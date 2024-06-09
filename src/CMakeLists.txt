ADD_LIBRARY(Phantom SHARED
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

target_compile_features(Phantom PUBLIC cxx_std_17 cuda_std_17)

set(PHANTOM_HEADER_FILES
        ../include/phantom.h

        ../include/batchencoder.h
        ../include/ciphertext.h
        ../include/ckks.h
        ../include/common.h
        ../include/context.h
        ../include/util/encryptionparams.h
        ../include/evaluate.cuh
        ../include/fft.h
        ../include/galois.cuh
        ../include/gpucontext.h
        ../include/gputype.h
        ../include/lock.cuh
        ../include/util/modulus.h
        ../include/ntt.cuh
        ../include/plaintext.h
        ../include/polymath.cuh
        ../include/prng.cuh
        ../include/rns.cuh
        ../include/rns_base.cuh
        ../include/rns_bconv.cuh
        ../include/scalingvariant.cuh
        ../include/secretkey.h
        ../include/uintmath.cuh
        ../include/uintmodmath.cuh
        ../include/util.cuh

        ../include/util/blake2b.h
        ../include/util/blake2-impl.h
        ../include/util/common.h
        ../include/util/defines.h
        ../include/util/galois.h
        ../include/util/globals.h
        ../include/util/hash.h
        ../include/util/hestdparms.h
        ../include/util/ntt.h
        ../include/util/numth.h
        ../include/util/rns.h
        ../include/util/uintarith.h
        ../include/util/uintarithmod.h
        ../include/util/uintarithsmallmod.h
        ../include/util/uintcore.h
)
