#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "phantom.h"

namespace py = pybind11;

PYBIND11_MODULE(pyPhantom, m) {
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

    m.def("coeff_modulus_create", &phantom::arith::CoeffModulus::Create);

    py::class_<phantom::EncryptionParameters>(m, "params")
            .def(py::init<phantom::scheme_type>())
            .def("set_mul_tech", &phantom::EncryptionParameters::set_mul_tech)
            .def("set_poly_modulus_degree", &phantom::EncryptionParameters::set_poly_modulus_degree)
            .def("set_special_modulus_size", &phantom::EncryptionParameters::set_special_modulus_size)
            .def("set_galois_elts", &phantom::EncryptionParameters::set_galois_elts)
            .def("set_coeff_modulus", &phantom::EncryptionParameters::set_coeff_modulus);

    py::class_<PhantomContext>(m, "context")
            .def(py::init<phantom::EncryptionParameters &>());

    py::class_<PhantomSecretKey>(m, "secret_key")
            .def(py::init<>())
            .def("gen_secretkey", &PhantomSecretKey::gen_secretkey, py::arg(), py::arg("stream") = nullptr)
            .def("gen_publickey", &PhantomSecretKey::gen_publickey)
            .def("gen_relinkey", &PhantomSecretKey::gen_relinkey)
            .def("create_galois_keys", &PhantomSecretKey::create_galois_keys)
            .def("encrypt_symmetric", py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(
                    &PhantomSecretKey::encrypt_symmetric, py::const_))
            .def("decrypt",
                 py::overload_cast<const PhantomContext &, const PhantomCiphertext &>(&PhantomSecretKey::decrypt));

    py::class_<PhantomPublicKey>(m, "public_key")
            .def(py::init<>())
            .def("encrypt_asymmetric", py::overload_cast<const PhantomContext &, const PhantomPlaintext &, const >(
                    &PhantomPublicKey::encrypt_asymmetric));

    py::class_<PhantomRelinKey>(m, "relin_key")
            .def(py::init<>());

    py::class_<PhantomGaloisKey>(m, "galois_key")
            .def(py::init<>());

    m.def("get_elts_from_steps", &get_elts_from_steps);

    py::class_<PhantomCKKSEncoder>(m, "ckks_encoder")
            .def(py::init<PhantomContext &>())
            .def("slot_count", &PhantomCKKSEncoder::slot_count)
            .def("encode", py::overload_cast<const PhantomContext &, const std::vector<double> &, double>(
                    &PhantomCKKSEncoder::encode))
            .def("encode_to", py::overload_cast<const PhantomContext &, const std::vector<double> &, size_t, double>(
                    &PhantomCKKSEncoder::encode))
            .def("decode",
                 py::overload_cast<const PhantomContext &, const PhantomPlaintext &>(&PhantomCKKSEncoder::decode));

    py::class_<PhantomPlaintext>(m, "plaintext")
            .def(py::init<>());

    py::class_<PhantomCiphertext>(m, "ciphertext")
            .def(py::init<>())
            .def("set_scale", &PhantomCiphertext::set_scale);

    m.def("negate_inplace", &negate_inplace);

    m.def("add_inplace", &add_inplace);
    m.def("add_plain_inplace", &add_plain_inplace);
    m.def("add_many", &add_many);

    m.def("sub_inplace", &sub_inplace, py::arg(), py::arg(), py::arg(), py::arg("negate") = false);
    m.def("sub_plain_inplace", &sub_plain_inplace);

    m.def("multiply_inplace", &multiply_inplace);
    m.def("multiply_and_relin_inplace", &multiply_and_relin_inplace);
    m.def("multiply_plain_inplace", &multiply_plain_inplace);

    m.def("relinearize_inplace", &relinearize_inplace);

    m.def("mod_switch_to_inplace",
          py::overload_cast<const PhantomContext &, PhantomPlaintext &, size_t>(&mod_switch_to_inplace));
    m.def("mod_switch_to_inplace",
          py::overload_cast<const PhantomContext &, PhantomCiphertext &, size_t>(&mod_switch_to_inplace));
    m.def("mod_switch_to_next",
          py::overload_cast<const PhantomContext &, const PhantomCiphertext &, PhantomCiphertext
          &>(&mod_switch_to_next));
    m.def("mod_switch_to_next_inplace",
          py::overload_cast<const PhantomContext &, PhantomPlaintext &>(&mod_switch_to_next_inplace));
    m.def("mod_switch_to_next_inplace",
          py::overload_cast<const PhantomContext &, PhantomCiphertext &>(&mod_switch_to_next_inplace));
    m.def("rescale_to_next", &rescale_to_next);
    m.def("rescale_to_next_inplace", &rescale_to_next_inplace);

    m.def("apply_galois_inplace", &apply_galois_inplace);

    m.def("rotate_rows_inplace", &rotate_rows_inplace);
    m.def("rotate_columns_inplace", &rotate_columns_inplace);
    m.def("rotate_vector_inplace", &rotate_vector_inplace);

    m.def("hoisting", &hoisting);
    m.def("hoisting_inplace", &hoisting_inplace);

    m.def("complex_conjugate_inplace", &complex_conjugate_inplace);
}
