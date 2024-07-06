include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(Phantom SHARED
        context.cu
        modulus.cu
        batchencoder.cu
        prng.cu
        polymath.cu
        secretkey.cu
        rns.cu
        rns_base.cu
        rns_bconv.cu
        scalingvariant.cu
        evaluate.cu
        fft.cu
        ckks.cu
        galois.cu
        eval_key_switch.cu

        ntt/fntt_2d.cu
        ntt/ntt_1d.cu
        ntt/ntt_keyswitch_old.cu
        ntt/intt_2d.cu
        ntt/ntt_modup.cu
        ntt/ntt_moddown.cu

        util/blake2b.cu
        util/blake2xb.cu
        util/globals.cu
        util/galois.cu
        util/hash.cu
        util/ntt.cu
        util/numth.cu
        util/rns.cu
        util/uintarith.cu
        util/uintarithmod.cu
        util/uintarithsmallmod.cu
)

target_compile_options(Phantom PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_features(Phantom PUBLIC cxx_std_17 cuda_std_17)
