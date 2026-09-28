#include <TensorInterop.hpp>
#include <atomic>
namespace py=pybind11;
using namespace nelux::tensor;
std::atomic<int64_t> deleted{0};
PYBIND11_MODULE(abi_probe,m) {
 nelux::python::initializeTensorInterop();
 m.def("identity",[](Tensor t){return t;});
 m.def("mul_i",&mul_integer); m.def("mul_f",&mul_floating);
 m.def("div_i",&div_integer); m.def("div_f",&div_floating);
 m.def("clamp",&nelux::tensor::clamp); m.def("round",&nelux::tensor::round);
 m.def("owned",[]{auto *p=new uint8_t[6]{1,2,3,4,5,6};return from_blob(p,{2,3},[](void *p){++deleted;delete[]static_cast<uint8_t*>(p);},kUInt8);});
 m.def("deleted",[]{return deleted.load();});
 m.def("empty",[]{return empty({2,3},kUInt16);});
 m.def("undefined",[]{return Tensor();});
 m.def("stack",[](Tensor t){return stack({t,t},0);});
 m.def("permute",[](Tensor t){return permute(t,{1,0});});
 m.def("arange",[](int64_t end){return arange(end);});
 m.def("bad_blob",[]{auto *p=new uint8_t[6];try{from_blob(p,{-1},[](void*p){++deleted;delete[]static_cast<uint8_t*>(p);},kUInt8);}catch(const std::exception&){};return deleted.load();});
}
