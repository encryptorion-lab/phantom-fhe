#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phantom.h"

namespace py = pybind11;

PYBIND11_MODULE(pyPhantom, m) {

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

    py::enum_<phantom::scheme_type>(m, "scheme_type")
            .value("none", phantom::scheme_type::none)
            .value("bgv", phantom::scheme_type::bgv)
            .value("bfv", phantom::scheme_type::bfv)
            .value("ckks", phantom::scheme_type::ckks)
            .export_values();

    py::enum_<phantom::mul_tech_type>(m, "mul_tech_type")
            .value("none", phantom::mul_tech_type::none)
            .value("behz", phantom::mul_tech_type::behz)
            .value("hps", phantom::mul_tech_type::hps)
            .value("hps_overq", phantom::mul_tech_type::hps_overq)
            .value("hps_overq_leveled", phantom::mul_tech_type::hps_overq_leveled)
            .export_values();

    py::enum_<phantom::arith::sec_level_type>(m, "sec_level_type")
            .value("none", phantom::arith::sec_level_type::none)
            .value("tc128", phantom::arith::sec_level_type::tc128)
            .value("tc192", phantom::arith::sec_level_type::tc192)
            .value("tc256", phantom::arith::sec_level_type::tc256)
            .export_values();

    py::class_<phantom::arith::Modulus>(m, "modulus")
            .def(py::init<std::uint64_t>());

    m.def("create_coeff_modulus", &phantom::arith::CoeffModulus::Create);

    m.def("create_plain_modulus", &phantom::arith::PlainModulus::Batching);

    py::class_<phantom::EncryptionParameters>(m, "params")
            .def(py::init<phantom::scheme_type>())
            .def("set_mul_tech", &phantom::EncryptionParameters::set_mul_tech)
            .def("set_poly_modulus_degree", &phantom::EncryptionParameters::set_poly_modulus_degree)
            .def("set_special_modulus_size", &phantom::EncryptionParameters::set_special_modulus_size)
            .def("set_galois_elts", &phantom::EncryptionParameters::set_galois_elts)
            .def("set_coeff_modulus", &phantom::EncryptionParameters::set_coeff_modulus)
            .def("set_plain_modulus", &phantom::EncryptionParameters::set_plain_modulus);

    py::class_<phantom::util::cuda_stream_wrapper>(m, "cuda_stream")
            .def(py::init<>());

    py::class_<PhantomContext>(m, "context")
            .def(py::init<phantom::EncryptionParameters &>());

    py::class_<PhantomSecretKey>(m, "secret_key")
            .def(py::init<const PhantomContext &>())
            .def("gen_publickey", &PhantomSecretKey::gen_publickey)
            .def("gen_relinkey", &PhantomSecretKey::gen_relinkey)
            .def("create_galois_keys", &PhantomSecretKey::create_galois_keys)
            .def("encrypt_symmetric",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &, const phantom::util::cuda_stream_wrapper &>(
                         &PhantomSecretKey::encrypt_symmetric, py::const_), py::arg(), py::arg(),
                 py::arg("stream") = *phantom::util::global_variables::default_stream)
            .def("decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &, const phantom::util::cuda_stream_wrapper &>(
                         &PhantomSecretKey::decrypt), py::arg(), py::arg(),
                 py::arg("stream") = *phantom::util::global_variables::default_stream);

    py::class_<PhantomPublicKey>(m, "public_key")
            .def(py::init<>())
            .def("encrypt_asymmetric",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &, const phantom::util::cuda_stream_wrapper &>(
                         &PhantomPublicKey::encrypt_asymmetric), py::arg(), py::arg(),
                 py::arg("stream") = *phantom::util::global_variables::default_stream);

    py::class_<PhantomRelinKey>(m, "relin_key")
            .def(py::init<>());

    py::class_<PhantomGaloisKey>(m, "galois_key")
            .def(py::init<>());

    m.def("get_elts_from_steps", &get_elts_from_steps);

    py::class_<PhantomBatchEncoder>(m, "batch_encoder")
            .def(py::init<const PhantomContext &>())
            .def("slot_count", &PhantomBatchEncoder::slot_count)
            .def("encode",
                 py::overload_cast<const PhantomContext &, const std::vector<uint64_t> &, const phantom::util::cuda_stream_wrapper &>(
                         &PhantomBatchEncoder::encode, py::const_),
                 py::arg(), py::arg(),
                 py::arg("stream") = *phantom::util::global_variables::default_stream)
            .def("decode",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &, const phantom::util::cuda_stream_wrapper &>(
                         &PhantomBatchEncoder::decode, py::const_),
                 py::arg(), py::arg(),
                 py::arg("stream") = *phantom::util::global_variables::default_stream);

    py::class_<PhantomCKKSEncoder>(m, "ckks_encoder")
            .def(py::init<const PhantomContext &>())
            .def("slot_count", &PhantomCKKSEncoder::slot_count)
            .def("encode_complex_vector",
                 py::overload_cast<const PhantomContext &, const std::vector<cuDoubleComplex> &, double, size_t, const phantom::util::cuda_stream_wrapper &>(
                         &PhantomCKKSEncoder::encode<cuDoubleComplex>),
                 py::arg(), py::arg(), py::arg(),
                 py::arg("chain_index") = 1,
                 py::arg("stream") = *phantom::util::global_variables::default_stream)
            .def("encode_double_vector",
                 py::overload_cast<const PhantomContext &, const std::vector<double> &, double, size_t, const phantom::util::cuda_stream_wrapper &>(
                         &PhantomCKKSEncoder::encode<double>),
                 py::arg(), py::arg(), py::arg(),
                 py::arg("chain_index") = 1,
                 py::arg("stream") = *phantom::util::global_variables::default_stream)
            .def("decode_complex_vector",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &, const phantom::util::cuda_stream_wrapper &>(
                         &PhantomCKKSEncoder::decode<cuDoubleComplex>),
                 py::arg(), py::arg(),
                 py::arg("stream") = *phantom::util::global_variables::default_stream)
            .def("decode_double_vector",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &, const phantom::util::cuda_stream_wrapper &>(
                         &PhantomCKKSEncoder::decode<double>),
                 py::arg(), py::arg(),
                 py::arg("stream") = *phantom::util::global_variables::default_stream);

    py::class_<PhantomPlaintext>(m, "plaintext")
            .def(py::init<>());

    py::class_<PhantomCiphertext>(m, "ciphertext")
            .def(py::init<>())
            .def("set_scale", &PhantomCiphertext::set_scale);

    m.def("negate", &negate, py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("add", &add, py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("add_plain", &add_plain, py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("add_many", &add_many, py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("sub", &sub, py::arg(), py::arg(), py::arg(),
          py::arg("negate") = false,
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("sub_plain", &sub_plain, py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("multiply", &multiply, py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("multiply_and_relin", &multiply_and_relin, py::arg(), py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("multiply_plain", &multiply_plain, py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("relinearize", &relinearize, py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomPlaintext &,
                  const phantom::util::cuda_stream_wrapper &>(&mod_switch_to_next), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &,
                  const phantom::util::cuda_stream_wrapper &>(&mod_switch_to_next), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("mod_switch_to", py::overload_cast<const PhantomContext &, const PhantomPlaintext &, size_t,
                  const phantom::util::cuda_stream_wrapper &>(&mod_switch_to), py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("mod_switch_to", py::overload_cast<const PhantomContext &, const PhantomCiphertext &, size_t,
                  const phantom::util::cuda_stream_wrapper &>(&mod_switch_to), py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("apply_galois", &apply_galois, py::arg(), py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("rotate_rows", &rotate_rows, py::arg(), py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("rotate_columns", &rotate_columns, py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("rotate_vector", &rotate_vector, py::arg(), py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("complex_conjugate", &complex_conjugate, py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);

    m.def("hoisting", &hoisting, py::arg(), py::arg(), py::arg(), py::arg(),
          py::arg("stream") = *phantom::util::global_variables::default_stream);
}
