/* Copyright (c) 2013-2015, EPFL/Blue Brain Project
 *                          Daniel Nachbaur <daniel.nachbaur@epfl.ch>
 *                          Juan Hernando <jhernando@fi.upm.es>
 *
 * This file is part of Brion <https://github.com/BlueBrain/Brion>
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License version 3.0 as published
 * by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include "circuit.h"

#include <bitset>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/scoped_ptr.hpp>
#include <lunchbox/stdExt.h>
#include <lunchbox/log.h>
#include <mvd/mvd3.hpp>

namespace brion
{

enum Section
{
    SECTION_NEURONS,
    SECTION_MICROBOX,
    SECTION_LAYERS,
    SECTION_SYNAPSES,
    SECTION_ETYPES,
    SECTION_MTYPES,
    SECTION_MCPOSITIONS,
    SECTION_CIRCUITSEEDS,
    SECTION_UNKNOWN
};

class Circuit::Impl{
public:
  virtual ~Impl(){}

  virtual NeuronMatrix get( const GIDSet& gids, const uint32_t attributes ) const =0;

  virtual Vector3fs getPositions( const GIDSet& gids ) const =0;

  virtual size_t getNumNeurons() const =0;

  virtual Strings getTypes( const NeuronClass type ) const = 0;
};


//////////////////////////////////////////////////
/// MVD2 implementation
//////////////////////////////////////////////////

class Circuit::ImplMVD2 : public Circuit::Impl
{
public:
    explicit ImplMVD2( const std::string& source )
    {
        typedef stde::hash_map< std::string, Section > LookUp;
        LookUp sections;
        sections.insert( std::make_pair( "Neurons Loaded", SECTION_NEURONS ));
        sections.insert( std::make_pair( "MicroBox Data", SECTION_MICROBOX ));
        sections.insert( std::make_pair( "Layers Positions Data",
                                         SECTION_LAYERS ));
        sections.insert( std::make_pair( "Axon-Dendrite Synapses",
                                         SECTION_SYNAPSES ));
        sections.insert( std::make_pair( "ElectroTypes", SECTION_ETYPES ));
        sections.insert( std::make_pair( "MorphTypes", SECTION_MTYPES ));
        sections.insert( std::make_pair( "MiniColumnsPosition",
                                         SECTION_MCPOSITIONS ));
        sections.insert( std::make_pair( "CircuitSeeds",
                                         SECTION_CIRCUITSEEDS ));

        try{
            _file.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
            _file.open( source.c_str( ));
            _file >> std::ws;
            Section current = SECTION_UNKNOWN;
            while( !_file.eof( ))
            {
                std::string content;
                std::getline( _file, content );
                _file >> std::ws;

                LookUp::const_iterator it = sections.find( content );
                if( it != sections.end() )
                    current = it->second;
                else
                    _table[current].push_back( content );
            }
        }catch(std::ifstream::failure & e){
            throw std::runtime_error(std::string("Impossible to Open/Read MVD2 file ") + source + " " + e.what());
        }
    }

    virtual ~ImplMVD2(){}

    virtual NeuronMatrix get( const GIDSet& gids, const uint32_t attributes ) const
    {
        const std::bitset< NEURON_ALL > bits( attributes );
        if( !bits.any( ))
            return NeuronMatrix();

        const Strings& neurons = _table.find( SECTION_NEURONS )->second;

        std::vector<int32_t> indices;
        indices.reserve( gids.size( ));
        BOOST_FOREACH( const uint32_t gid, gids)
        {
            if ( gid > neurons.size( ))
            {
                std::stringstream msg;
                msg << "Cell GID out of range: " << gid;
                throw std::runtime_error( msg.str().c_str( ));
            }
            indices.push_back( gid - 1 );
        }

        const size_t numNeurons =
            indices.empty() ? getNumNeurons() : indices.size();
        NeuronMatrix values( boost::extents[numNeurons][bits.count()] );
        std::vector<char> buffer;

        // This loop uses a hand-written string tokenizer to improve the
        // performance of loading very large circuits (millions of neurons).
        // This code is faster than using boost tokenizer because it does not
        // use std::string and does only very few memory allocations.
        for( size_t i = 0; i < numNeurons; ++i )
        {
            const size_t neuronIdx = indices.empty() ? i : indices[i];

            const std::string& line = neurons[neuronIdx];
            buffer.resize( std::max( buffer.size(), line.size() + 1 ));
            buffer[0] = 0;
            size_t bit = 0;
            size_t field = 0;
            size_t k = 0;
            for( unsigned int j = 0; j != line.size() + 1; ++j)
            {
                char c = line[j];
                if( c == ' ' || c == 0)
                {
                    // A new input field is stored in buffer (except for the
                    // null terminating character.
                    if( bits.test( bit ))
                    {
                        buffer[k] = 0;
                        values[i][field++] = buffer.data();
                    }
                    ++bit;
                    k = 0;
                    buffer[0] = 0;

                    // Skipping white space and stopping j at the position
                    // of the next character.
                    while( c == ' ' )
                        c = line[++j];
                }
                buffer[k++] = c;
            }
        }
        return values;
    }

    virtual Vector3fs getPositions( const GIDSet& gids ) const{

        const NeuronMatrix& data = get(
            gids, brion::NEURON_POSITION_X | brion::NEURON_POSITION_Y |
                 brion::NEURON_POSITION_Z );

        GIDSet::const_iterator gid = gids.begin();
        Vector3fs positions( gids.size( ));
        #pragma omp parallel for
        for( size_t i = 0; i < gids.size(); ++i )
        {
            try
            {
                positions[i] =
                    brion::Vector3f( boost::lexical_cast< float >( data[i][0] ),
                                     boost::lexical_cast< float >( data[i][1] ),
                                     boost::lexical_cast< float >( data[i][2] ));
            }
            catch( const boost::bad_lexical_cast& )
            {
                LBWARN << "Error parsing circuit position or orientation for gid "
                       << *gid << ". Morphology not transformed." << std::endl;
                positions[i] = Vector3f::ZERO;
            }
            #pragma omp critical (brain_circuit_getPositions)
            ++gid;
        }
        return positions;

    }

    virtual size_t getNumNeurons() const
    {
        return _table.find( SECTION_NEURONS )->second.size();
    }

    virtual Strings getTypes( const NeuronClass type ) const
    {
        switch( type )
        {
        case NEURONCLASS_ETYPE:{
            CircuitTable::const_iterator i = _table.find( SECTION_ETYPES );
            return i != _table.end() ? i->second : Strings();
        }

        case NEURONCLASS_MTYPE:{
        case NEURONCLASS_MORPHOLOGY_CLASS:
        case NEURONCLASS_FUNCTION_CLASS:
        {
            CircuitTable::const_iterator i = _table.find( SECTION_MTYPES );
            if( i == _table.end( ))
                return Strings();
            const Strings& data = i->second;
            Strings buffer( data.size( ));
            typedef boost::tokenizer< boost::char_separator< char > > tokenizer;
            boost::char_separator< char > sep( " " );
            for( size_t j = 0; j < buffer.size(); ++j )
            {
                const tokenizer tokens( data[j], sep );
                tokenizer::const_iterator t = tokens.begin();
                std::advance( t, int32_t(type));
                buffer[j] = *t;
            }
            return buffer;
        }

        case NEURONCLASS_INVALID:
        default:
            return Strings();
        }
        }
    }

private:
    std::ifstream _file;

    typedef stde::hash_map< uint32_t, Strings > CircuitTable;
    CircuitTable _table;
};


//////////////////////////////////////////////////
/// MVD3 implementation
//////////////////////////////////////////////////

template<typename SrcArray, typename DstArray, typename AssignOps>
inline void array_range_to_index(const MVD3::Range & range,
                                 const GIDSet& gids,
                                 SrcArray & src,
                                 DstArray & dst,
                                 const AssignOps & assign_ops){
    size_t s_source = std::distance(src.begin(), src.end());
    size_t s_dest = std::distance(dst.begin(), dst.end());

    if(gids.empty() == true){ // we work on full range, no translation needed
        assert(s_source == s_dest);
        std::transform(src.begin(), src.end(), dst.begin(), assign_ops);
    }else{
        assert(s_dest == gids.size());
        typename DstArray::iterator dst_it = dst.begin();
        for(GIDSet::const_iterator it = gids.begin(); it != gids.end(); ++it){
            assert( *it >= range.offset && *it < (range.offset + range.count+1));
            typename SrcArray::iterator src_it = src.begin();
            std::advance(src_it, *it - range.offset );
            *dst_it = assign_ops(*src_it);
            ++dst_it;
        }
    }
}

std::string swap_string(std::string & in){
    std::string res;
    res.swap(in);
    return res;
}

std::string size_to_string(size_t in){
    return boost::lexical_cast<std::string>(in);
}

Vector3f boost_mul_array_to_vmml(const MVD3::Positions::subarray<1>::type & subarray){
    Vector3f res;
    std::copy(subarray.begin(), subarray.end(), res.begin());
    return res;
}


MVD3::Range range_from_gidset(const GIDSet & gids){
    if(gids.size() > 0){
        const size_t offset = *gids.begin();
        const size_t range_count = *gids.rbegin() - offset +1;
        return MVD3::Range(offset, range_count);
    }
    return MVD3::Range(0,0); // full range
}

class Circuit::ImplMVD3 : public Circuit::Impl
{
public:
    ImplMVD3(const std::string & src) : _file(){
        // convert any HighFive exception
        try{
            _file.reset(new MVD3::MVD3File(src));
        }catch(std::exception & e){
            throw std::runtime_error(std::string("Impossible to Open/Read MVD3 file ") + src + " " + e.what());
        }
    }

    virtual ~ImplMVD3() {}

    virtual NeuronMatrix get( const GIDSet& gids, const uint32_t attributes ) const{
        NeuronMatrix result;
        std::bitset< NEURON_ALL > bits( attributes );
        MVD3::Range range = range_from_gidset(gids);

        if(bits.none()){
            return result;
        }

        const size_t size_result = (gids.size() > 0)?gids.size():getNumNeurons();
        result.resize(boost::extents[size_result][bits.count()]);

        size_t pos =0;
        for(int i=0; i < NEURON_ALL; ++i){
            typedef NeuronMatrix::index_range NeuronMatrixRange;
            if(bits[i] == true){
                switch(1 << i){
                case NEURON_COLUMN_GID:
                case NEURON_MINICOLUMN_GID:
                case NEURON_LAYER:
                case NEURON_METYPE:{
                    throw std::runtime_error("Layer, metype, mini-column or column gid informations are not supported by the MVD3 file format");
                }
                case NEURON_ROTATION:
                case NEURON_POSITION_X:
                case NEURON_POSITION_Y:
                case NEURON_POSITION_Z:{
                    throw std::runtime_error("Position (X,Y,Z) or rotation informations are not string properties in MVD3"
                                             "please see Brion::Circuit API for details");
                }
                case NEURON_ETYPE:{
                    std::vector<size_t> etypes = _file->getIndexEtypes(range);
                    NeuronMatrix::array_view<1>::type column_view = result[boost::indices[NeuronMatrixRange()][pos]];
                    array_range_to_index(range, gids, etypes, column_view, size_to_string);
                    break;
                }
                case NEURON_MTYPE:{
                    std::vector<size_t> mtypes = _file->getIndexMtypes(range);
                    NeuronMatrix::array_view<1>::type column_view = result[boost::indices[NeuronMatrixRange()][pos]];
                    array_range_to_index(range, gids, mtypes, column_view, size_to_string);
                    break;
                }
                case NEURON_MORPHOLOGY_NAME:{
                    std::vector<std::string> morphos = _file->getMorphologies(range);
                    NeuronMatrix::array_view<1>::type column_view = result[boost::indices[NeuronMatrixRange()][pos]];
                    array_range_to_index(range, gids, morphos, column_view, swap_string);
                    break;
                }
                default:
                break;
                }
                pos += 1;
            }
        }
        assert(pos == bits.count());
        return result;
    }

    virtual Vector3fs getPositions(const GIDSet &gids) const{
        Vector3fs results;
        MVD3::Range range = range_from_gidset(gids);
        MVD3::Positions positions = _file->getPositions(range);
        assert(positions.shape()[1] ==3);
        array_range_to_index(range, gids, positions, results, boost_mul_array_to_vmml);
        return results;
    }

    virtual size_t getNumNeurons() const{
        return _file->getNbNeuron();
    }


    virtual Strings getTypes( const NeuronClass type ) const{
        switch( type ){
            case NEURONCLASS_ETYPE:{
                return _file->listAllEtypes();
            }
            case NEURONCLASS_MTYPE:{
                return _file->listAllMtypes();
            }
            case NEURONCLASS_MORPHOLOGY_CLASS:{
                return _file->listAllMorphologies();
            }
            case NEURONCLASS_FUNCTION_CLASS:{
                return _file->listAllSynapseClass();
            }
            case NEURONCLASS_INVALID:
            default:
            break;
        }
        return Strings();
    }

private:
    mutable boost::scoped_ptr<MVD3::MVD3File> _file;
};


//////////////////////////////////////////////////
/// Circuit Pimpl mapper
//////////////////////////////////////////////////

Circuit::Impl* Circuit::instanciate(const std::string & source){
    namespace fs = boost::filesystem;
    fs::path path = source;
    const std::string ext = fs::extension( path );
    if( ext == ".mvd" || ext == ".mvd2" ){
        return new Circuit::ImplMVD2(source);
    }
    if( ext == ".mvd3"){
       return new Circuit::ImplMVD3(source);
    }
    throw std::runtime_error( "Expecting mvd file format for circuit "
                                  "file " + source );
}

Circuit::Circuit( const std::string& source )
    : _impl(instanciate(source))
{
}

Circuit::Circuit( const URI& source )
    : _impl( instanciate( source.getPath( )))
{
}

Circuit::~Circuit()
{
    delete _impl;
}

NeuronMatrix Circuit::get( const GIDSet& gids, const uint32_t attributes ) const
{
    return _impl->get( gids, attributes );
}

size_t Circuit::getNumNeurons() const
{
    return _impl->getNumNeurons();
}

Vector3fs Circuit::getPositions( const GIDSet& gids ) const{
    return _impl->getPositions(gids);
}

Strings Circuit::getTypes( const NeuronClass type ) const
{
    return _impl->getTypes( type );
}

}
