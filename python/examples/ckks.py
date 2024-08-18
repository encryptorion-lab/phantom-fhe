import pyPhantom as phantom

log_n = 14
n = 2 ** log_n
modulus_chain = [60, 40, 40, 60]
galois_steps = [1, 2, 3, 4, 5, 6, 7]
size_P = 1
scale = 2.0 ** 40

params = phantom.params(phantom.scheme_type.ckks)
params.set_poly_modulus_degree(n)
params.set_coeff_modulus(phantom.create_coeff_modulus(n, modulus_chain))
params.set_special_modulus_size(size_P)
params.set_galois_elts(phantom.get_elts_from_steps(galois_steps, n))

context = phantom.context(params)

sk = phantom.secret_key(context)
pk = sk.gen_publickey(context)
rlk = sk.gen_relinkey(context)
glk = sk.create_galois_keys(context)

encoder = phantom.ckks_encoder(context)
slot_count = encoder.slot_count()
print("slot_count", slot_count)

msg = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
msg += [0.0] * (slot_count - len(msg))

pt = encoder.encode_double_vector(context, msg, scale, chain_index=1)
ct = pk.encrypt_asymmetric(context, pt)

ct = phantom.multiply_and_relin(context, ct, ct, rlk)
ct = phantom.rescale_to_next(context, ct)

ct2 = phantom.hoisting(context, ct, glk, [1, 2, 3, 4, 5, 6, 7])
ct = phantom.add(context, ct, ct2)

pt_dec = sk.decrypt(context, ct)
result = encoder.decode_double_vector(context, pt_dec)

formatted_result = ['%.3f' % ele for ele in result]
print(formatted_result[:8])
