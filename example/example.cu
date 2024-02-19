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
        cout << "| 1. BFV Basics              | 1_bfv_basics.cu            |" << endl;
        cout << "| 2. Encoders                | 2_encoders.cu              |" << endl;
        cout << "| 3. BGV Basics              | 3_bgv_basics.cu            |" << endl;
        cout << "| 4. CKKS Basics             | 4_ckks_basics.cu           |" << endl;
        cout << "| 5. BFV Opt                 | 5_bfv_opt.cu               |" << endl;
        cout << "| 6. Kernel Fusing           | 6_kernel_fusing.cu         |" << endl;
        cout << "+----------------------------+----------------------------+" << endl;

        int selection = 0;
        bool valid = true;
        do {
            cout << endl << "> Run example (1 ~ 6) or exit (0): ";
            if (!(cin >> selection)) {
                valid = false;
            }
            else if (selection < 0 || selection > 6) {
                valid = false;
            }
            else {
                valid = true;
            }
            if (!valid) {
                cout << "  [Beep~~] valid option: type 0 ~ 7" << endl;
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
        }
        while (!valid);

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
                break;

            case 2:
                example_encoders();
                break;

            case 3:
                examples_bgv();
                break;

            case 4:
                examples_ckks();
                break;

            case 5:
                example_bfv_encrypt_decrypt_hps();
                example_bfv_encrypt_decrypt_hps_asym();
                example_bfv_hybrid_key_switching();
                example_bfv_multiply_correctness();
                example_bfv_multiply_benchmark();
                break;

            case 6:
                example_kernel_fusing();
                break;

            case 0:
            default:
                return 0;
        }
    }
}
