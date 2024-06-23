import pyPhantom as phantom

log_n = 14
n = 2 ** log_n
modulus_chain = [35, 35, 35, 35, 40, 40]
plain_modulus_bits = 20
galois_steps = [1]
size_P = 2

params = phantom.params(phantom.scheme_type.bgv)
params.set_poly_modulus_degree(n)
params.set_coeff_modulus(phantom.create_coeff_modulus(n, modulus_chain))
params.set_plain_modulus(phantom.create_plain_modulus(n, plain_modulus_bits))
params.set_special_modulus_size(size_P)
params.set_galois_elts(phantom.get_elts_from_steps(galois_steps, n))

context = phantom.context(params)

sk = phantom.secret_key(context)
pk = sk.gen_publickey(context)
rlk = sk.gen_relinkey(context)
glk = sk.create_galois_keys(context)

encoder = phantom.batch_encoder(context)
slot_count = encoder.slot_count()
print("slot_count", slot_count)

s = phantom.cuda_stream()


def bgv_test():
    msg = [1, 2, 3, 4, 5, 6, 7, 8]
    msg += [0] * (slot_count - len(msg))

    print('input:', msg[:8])

    pt = encoder.encode(context, msg, stream=s)
    ct = pk.encrypt_asymmetric(context, pt, stream=s)

    ct = phantom.multiply_and_relin(context, ct, ct, rlk, stream=s)
    ct = phantom.mod_switch_to_next(context, ct, stream=s)

    pt_dec = sk.decrypt(context, ct, stream=s)
    result = encoder.decode(context, pt_dec, stream=s)

    print('output:', result[:8])


bgv_test()
