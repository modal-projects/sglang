/* AES-256-GCM over batches of HiCache host pages.
 *
 * One call handles a whole batch without returning to Python, so worker
 * threads never contend with the scheduler for the interpreter lock.
 *
 * A pool is described by its host buffers. Page p owns the byte range
 * [base + p * stride, base + p * stride + size) of every buffer; those ranges,
 * concatenated in buffer order, are the page's byte stream. The stream is cut
 * into units of `unit_bytes`, each an independent AES-GCM message, so a large
 * page spreads across threads.
 *
 * The nonce of a unit is (pool, page, generation, unit). The generation of a
 * page increases every time the page is encrypted, so a nonce is never reused
 * under one key, and ciphertext that is moved to another page, or replayed
 * from an older generation, fails authentication.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* The part of OpenSSL's EVP interface used here, declared directly so that
 * building needs the OpenSSL runtime library but not its headers. */
typedef struct evp_cipher_ctx_st EVP_CIPHER_CTX;
typedef struct evp_cipher_st EVP_CIPHER;
typedef struct engine_st ENGINE;
EVP_CIPHER_CTX* EVP_CIPHER_CTX_new(void);
void EVP_CIPHER_CTX_free(EVP_CIPHER_CTX* ctx);
int EVP_CIPHER_CTX_ctrl(EVP_CIPHER_CTX* ctx, int type, int arg, void* ptr);
const EVP_CIPHER* EVP_aes_256_gcm(void);
int EVP_EncryptInit_ex(EVP_CIPHER_CTX*, const EVP_CIPHER*, ENGINE*, const unsigned char* key, const unsigned char* iv);
int EVP_DecryptInit_ex(EVP_CIPHER_CTX*, const EVP_CIPHER*, ENGINE*, const unsigned char* key, const unsigned char* iv);
int EVP_EncryptUpdate(EVP_CIPHER_CTX*, unsigned char* out, int* outl, const unsigned char* in, int inl);
int EVP_DecryptUpdate(EVP_CIPHER_CTX*, unsigned char* out, int* outl, const unsigned char* in, int inl);
int EVP_EncryptFinal_ex(EVP_CIPHER_CTX*, unsigned char* out, int* outl);
int EVP_DecryptFinal_ex(EVP_CIPHER_CTX*, unsigned char* out, int* outl);
#define EVP_CTRL_GCM_GET_TAG 0x10
#define EVP_CTRL_GCM_SET_TAG 0x11

#define HC_NONCE_BYTES 12
#define HC_TAG_BYTES 16
#define HC_MAX_UPDATE (1 << 30)

typedef struct {
  uint64_t base;
  uint64_t stride;
  uint64_t size;
} hc_buffer;

typedef struct {
  uint64_t page;
  uint64_t generation;
  uint32_t unit;
  uint32_t reserved;
} hc_item;

static void hc_nonce(unsigned char* nonce, uint32_t pool, const hc_item* item) {
  nonce[0] = (unsigned char)pool;
  for (int i = 0; i < 4; ++i) nonce[1 + i] = (unsigned char)(item->page >> (8 * i));
  for (int i = 0; i < 5; ++i) nonce[5 + i] = (unsigned char)(item->generation >> (8 * i));
  nonce[10] = (unsigned char)(item->unit);
  nonce[11] = (unsigned char)(item->unit >> 8);
}

/* Runs one unit of a page through the cipher in place. */
static int hc_unit(
    EVP_CIPHER_CTX* ctx,
    int encrypt,
    const hc_buffer* buffers,
    int num_buffers,
    uint64_t unit_bytes,
    const hc_item* item) {
  uint64_t skip = (uint64_t)item->unit * unit_bytes;
  uint64_t left = unit_bytes;
  int outlen = 0;
  for (int b = 0; b < num_buffers && left > 0; ++b) {
    uint64_t size = buffers[b].size;
    if (skip >= size) {
      skip -= size;
      continue;
    }
    unsigned char* p = (unsigned char*)(uintptr_t)(buffers[b].base + item->page * buffers[b].stride + skip);
    uint64_t n = size - skip < left ? size - skip : left;
    skip = 0;
    left -= n;
    while (n > 0) {
      int step = n > HC_MAX_UPDATE ? HC_MAX_UPDATE : (int)n;
      int ok = encrypt ? EVP_EncryptUpdate(ctx, p, &outlen, p, step) : EVP_DecryptUpdate(ctx, p, &outlen, p, step);
      if (ok != 1) return 0;
      p += step;
      n -= (uint64_t)step;
    }
  }
  return 1;
}

/* Encrypts units in place and stores each tag at
 * tags[(page * units_per_page + unit) * 16]. Returns 0 on success, or
 * 1 + the index of the item that failed. */
int hc_seal(
    const unsigned char* key,
    uint32_t pool,
    const hc_buffer* buffers,
    int num_buffers,
    uint64_t unit_bytes,
    uint32_t units_per_page,
    int64_t num_items,
    const hc_item* items,
    unsigned char* tags) {
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  unsigned char nonce[HC_NONCE_BYTES];
  unsigned char tail[32];
  int outlen = 0;
  int64_t failed = 0;
  if (ctx == NULL) return 1;
  if (EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, NULL) != 1) failed = 1;
  for (int64_t i = 0; i < num_items && !failed; ++i) {
    hc_nonce(nonce, pool, &items[i]);
    int ok = EVP_EncryptInit_ex(ctx, NULL, NULL, NULL, nonce) == 1 &&
             hc_unit(ctx, 1, buffers, num_buffers, unit_bytes, &items[i]) &&
             EVP_EncryptFinal_ex(ctx, tail, &outlen) == 1 &&
             EVP_CIPHER_CTX_ctrl(
                 ctx,
                 EVP_CTRL_GCM_GET_TAG,
                 HC_TAG_BYTES,
                 tags + (items[i].page * units_per_page + items[i].unit) * HC_TAG_BYTES) == 1;
    if (!ok) failed = i + 1;
  }
  EVP_CIPHER_CTX_free(ctx);
  return (int)failed;
}

/* Decrypts units in place. Returns the number of units that failed
 * authentication; `first_failed` receives the index of the first one. The
 * bytes of a unit that failed must not be used. */
int hc_open(
    const unsigned char* key,
    uint32_t pool,
    const hc_buffer* buffers,
    int num_buffers,
    uint64_t unit_bytes,
    uint32_t units_per_page,
    int64_t num_items,
    const hc_item* items,
    const unsigned char* tags,
    int64_t* first_failed) {
  EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
  unsigned char nonce[HC_NONCE_BYTES];
  unsigned char tag[HC_TAG_BYTES];
  unsigned char tail[32];
  int outlen = 0;
  int failures = 0;
  *first_failed = -1;
  if (ctx == NULL || EVP_DecryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, NULL) != 1) {
    if (ctx != NULL) EVP_CIPHER_CTX_free(ctx);
    *first_failed = 0;
    return (int)num_items;
  }
  for (int64_t i = 0; i < num_items; ++i) {
    hc_nonce(nonce, pool, &items[i]);
    memcpy(tag, tags + (items[i].page * units_per_page + items[i].unit) * HC_TAG_BYTES, HC_TAG_BYTES);
    int ok = EVP_DecryptInit_ex(ctx, NULL, NULL, NULL, nonce) == 1 &&
             hc_unit(ctx, 0, buffers, num_buffers, unit_bytes, &items[i]) &&
             EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_TAG, HC_TAG_BYTES, tag) == 1 &&
             EVP_DecryptFinal_ex(ctx, tail, &outlen) == 1;
    if (!ok) {
      if (failures == 0) *first_failed = i;
      ++failures;
    }
  }
  EVP_CIPHER_CTX_free(ctx);
  return failures;
}
