#pragma once
#ifndef DECODERS_HPP
#define DECODERS_HPP

#include <backends/Decoder.hpp>
#include <backends/cpu/Decoder.hpp>

#ifdef NELUX_ENABLE_CUDA
#include <backends/cuda/Decoder.hpp>
#endif

#endif // DECODERS_HPP
