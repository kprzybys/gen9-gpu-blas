// Opisy jąder i programów dla kcl (generuje gen_kernels.py z .ze_info). Offsety w danych wspólnych (cross-thread),
// -1 = brak. arg: off/size -- wartość albo adres stateless; wsk -- argument wskaźnikowy; bti -- indeks w binding table
// (stateful); off_adr / off_boff -- buffer_address / buffer_offset dla stateful.
#pragma once
#include <stddef.h>
#include <stdint.h>

struct kcl_arg_opis { int16_t off, size, wsk, bti, off_adr, off_boff; };

struct kcl_jadro_opis {
    const char *nazwa; const uint8_t *isa; uint32_t isa_len;
    uint32_t simd, cross, per_thread, lid, nargs;
    struct kcl_arg_opis arg[24];
    int16_t off_ls[3], off_els[3], off_gid[3], off_gs[3], off_ng[3], off_wdim, off_const, off_priv;
    uint32_t private_size, slm, bariera, nbti;
    const char *nieobslugiwane;   // "" = obsługiwane
};

struct kcl_program_opis {
    const char *plik, *opcje;     // opcje = NULL: binarka z asemblera (program_bin)
    const struct kcl_jadro_opis *jadra; uint32_t njader;
    const uint8_t *stale; uint32_t nstale;
};

extern const struct kcl_program_opis kcl_programy[];
extern const int kcl_nprogramow;
