import pyPhantom as phantom
from codetiming import Timer

log_n = 13
n = 2 ** log_n
modulus_chain = [60, 40, 40, 60]
galois_steps = [1, 2, 3, 4, 5, 6, 7]
size_P = 1
scale = 2.0 ** 40

params = phantom.params(phantom.scheme_type.ckks)
params.set_poly_modulus_degree(n)
params.set_coeff_modulus(phantom.coeff_modulus_create(n, modulus_chain))
params.set_special_modulus_size(size_P)
params.set_galois_elts(phantom.get_elts_from_steps(galois_steps, n))

context = phantom.context(params, True, phantom.sec_level_type.tc128)

sk = phantom.secret_key(params)
pk = sk.gen_publickey(context)
rlk = sk.gen_relinkey(context)
glk = sk.create_galois_keys(context)

encoder = phantom.ckks_encoder(context)
slot_count = encoder.slot_count()
msg = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
msg += [0.0] * (slot_count - len(msg))

pt = encoder.encode(context, msg, scale)
ct = pk.encrypt_asymmetric(context, pt)

hoisting_timer = Timer(name="hoisting", text="{name},{milliseconds:.3f} ms")
rotate_add_timer = Timer(name="rotate_add", text="{name},{milliseconds:.3f} ms")

for i in range(1000):
    hoisting_timer.start()
    ct2 = phantom.hoisting(context, ct, glk, [1, 2, 3, 4, 5, 6, 7])
    ct2 = phantom.add(context, ct2, ct)
    hoisting_timer.stop()

    rotate_add_timer.start()
    ct2 = phantom.rotate_vector(context, ct, 1, glk)
    ct2 = phantom.add(context, ct2, ct)
    ct4 = phantom.rotate_vector(context, ct2, 2, glk)
    ct4 = phantom.add(context, ct4, ct2)
    ct8 = phantom.rotate_vector(context, ct4, 4, glk)
    ct8 = phantom.add(context, ct8, ct4)
    rotate_add_timer.stop()

print("Hoisting mean time: ", Timer.timers.mean("hoisting") * 1000)
print("Rotate_add mean time: ", Timer.timers.mean("rotate_add") * 1000)

ct2 = phantom.hoisting(context, ct, glk, [1, 2, 3, 4, 5, 6, 7])
ct = phantom.add(context, ct, ct2)

pt_dec = sk.decrypt(context, ct)
result = encoder.decode(context, pt_dec)

formatted_result = ['%.1f' % ele for ele in result]
print(formatted_result[:8])
