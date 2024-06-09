#include <iostream>
#include "example.h"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc == 2) {
        auto selection = strtol(argv[1], nullptr, 10);
        if (selection == 1) {
            example_bfv_basics();
            example_bfv_batch_unbatch();
            example_bfv_encrypt_decrypt();
            example_bfv_encrypt_decrypt_asym();
            example_bfv_add();
            example_bfv_sub();
            example_bfv_mul();
            example_bfv_square();
            example_bfv_add_plain();
            example_bfv_sub_plain();
            example_bfv_mul_many_plain();
            example_bfv_mul_one_plain();
            example_bfv_rotate_column();
            example_bfv_rotate_row();

            example_bfv_encrypt_decrypt_hps();
            example_bfv_encrypt_decrypt_hps_asym();
            example_bfv_hybrid_key_switching();
            example_bfv_multiply_correctness();
            example_bfv_multiply_benchmark();
            return 0;
        }
        if (selection == 2) {
            examples_bgv();
            return 0;
        }
        if (selection == 3) {
            examples_ckks();
            return 0;
        }
        if (selection == 4) {
            example_kernel_fusing();
            return 0;
        }
        return 0;
    }

    while (true) {
        cout << "+---------------------------------------------------------+" << endl;
        cout << "| Examples                   | Source Files               |" << endl;
        cout << "+----------------------------+----------------------------+" << endl;
        cout << "| 1. BFV                     | 1_bfv.cu                   |" << endl;
        cout << "| 2. BGV                     | 2_bgv.cu                   |" << endl;
        cout << "| 3. CKKS                    | 3_ckks.cu                  |" << endl;
        cout << "| 4. Kernel Fusing           | 4_kernel_fusing.cu         |" << endl;
        cout << "+----------------------------+----------------------------+" << endl;

        int selection = 0;
        bool valid = true;
        do {
            cout << endl << "> Run example (1 ~ 4) or exit (0): ";
            if (!(cin >> selection)) {
                valid = false;
            } else if (selection < 0 || selection > 4) {
                valid = false;
            } else {
                valid = true;
            }
            if (!valid) {
                cout << "  [Beep~~] valid option: type 0 ~ 4" << endl;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
        } while (!valid);

        switch (selection) {
            case 1:
                example_bfv_basics();
                example_bfv_batch_unbatch();
                example_bfv_encrypt_decrypt();
                example_bfv_encrypt_decrypt_asym();
                example_bfv_add();
                example_bfv_sub();
                example_bfv_mul();
                example_bfv_square();
                example_bfv_add_plain();
                example_bfv_sub_plain();
                example_bfv_mul_many_plain();
                example_bfv_mul_one_plain();
                example_bfv_rotate_column();
                example_bfv_rotate_row();

                example_bfv_encrypt_decrypt_hps();
                example_bfv_encrypt_decrypt_hps_asym();
                example_bfv_hybrid_key_switching();
                example_bfv_multiply_correctness();
                example_bfv_multiply_benchmark();
                break;

            case 2:
                examples_bgv();
                break;

            case 3:
                examples_ckks();
                break;

            case 4:
                example_kernel_fusing();
                break;

            case 0:
            default:
                return 0;
        }
    }
}
