#pragma once

#include <iomanip>
#include <iostream>
#include "phantom.h"
#include "context.cuh"

#define EPSINON 0.001
#define complex_equ(a, b) ((fabs(a.x - b.x) < EPSINON) && (fabs(a.y - b.y) < EPSINON))

inline bool operator==(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) {
    return fabs(lhs.x - rhs.x) < EPSINON;
}

/*
Helper function: Prints the name of the example in a fancy banner.
*/
inline void print_example_banner(std::string title) {
    if (!title.empty()) {
        std::size_t title_length = title.length();
        std::size_t banner_length = title_length + 2 * 10;
        std::string banner_top = "+" + std::string(banner_length - 2, '-') + "+";
        std::string banner_middle = "|" + std::string(9, ' ') + title + std::string(9, ' ') + "|";

        std::cout << std::endl
                << banner_top << std::endl
                << banner_middle << std::endl
                << banner_top << std::endl;
    }
}

/*
Helper function: Prints the parameters in a SEALContext.
*/
inline void print_parameters(const PhantomContext &context) {
    auto &context_data = context.get_context_data(0);
    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme()) {
        case phantom::scheme_type::bfv:
            scheme_name = "BFV";
            break;
        case phantom::scheme_type::ckks:
            scheme_name = "CKKS";
            break;
        case phantom::scheme_type::bgv:
            scheme_name = "BGV";
            break;
        default:
            throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++) {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;

    std::cout << std::endl;
    for (std::size_t i = 0; i < coeff_modulus_size; i++) {
        std::cout << coeff_modulus[i].value() << " ,  ";
    }
    std::cout << std::endl
            << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == phantom::scheme_type::bfv) {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    std::cout << "\\" << std::endl;
}

template<typename T>
inline void print_vector(std::vector<T> vec, std::size_t print_size = 4, int prec = 3) {
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    std::size_t slot_count = vec.size();

    std::cout << std::fixed << std::setprecision(prec);
    std::cout << std::endl;
    if (slot_count <= 2 * print_size) {
        std::cout << "    [";
        for (std::size_t i = 0; i < slot_count; i++) {
            std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    else {
        vec.resize(std::max(vec.size(), 2 * print_size));
        std::cout << "    [";
        for (std::size_t i = 0; i < print_size; i++) {
            std::cout << " " << vec[i] << ",";
        }
        if (vec.size() > 2 * print_size) {
            std::cout << " ...,";
        }
        for (std::size_t i = slot_count - print_size; i < slot_count; i++) {
            std::cout << " " << vec[i] << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

/*
Helper function: Prints a vector of floating-point values.
*/
inline void print_vector(std::vector<cuDoubleComplex> vec, std::size_t print_size = 4, int prec = 3) {
    /*
    Save the formatting information for std::cout.
    */
    std::ios old_fmt(nullptr);
    old_fmt.copyfmt(std::cout);

    std::size_t slot_count = vec.size();

    std::cout << std::fixed << std::setprecision(prec);
    std::cout << std::endl;
    if (slot_count <= 2 * print_size) {
        std::cout << "    [";
        for (std::size_t i = 0; i < slot_count; i++) {
            std::cout << " " << vec[i].x << " + i * " << vec[i].y << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    else {
        vec.resize(std::max(vec.size(), 2 * print_size));
        std::cout << "    [";
        for (std::size_t i = 0; i < print_size; i++) {
            std::cout << " " << vec[i].x << " + i * " << vec[i].y << ",";
        }
        if (vec.size() > 2 * print_size) {
            std::cout << " ...,";
        }
        for (std::size_t i = slot_count - print_size; i < slot_count; i++) {
            std::cout << " " << vec[i].x << " + i * " << vec[i].y << ((i != slot_count - 1) ? "," : " ]\n");
        }
    }
    std::cout << std::endl;

    /*
    Restore the old std::cout formatting.
    */
    std::cout.copyfmt(old_fmt);
}

/*
Helper function: Print line number.
*/
inline void print_line(int line_number) {
    std::cout << "Line " << std::setw(3) << line_number << " --> ";
}

