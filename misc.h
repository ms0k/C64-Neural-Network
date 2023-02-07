#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__C64__)
#include <6502.h>
#include <_6526.h>
#include <_vic2.h>
#include <c64.h>

/*Preferring to access some registers via macro for now (like BASIC V2 POKE,
 * PEEK)*/
#define JMP(x) ((void (*)(void))(x))()
#define ADDR(x) (*(volatile uint8_t *)(x))

/*Generated neural network code entry point*/
void entry(const int8_t[1][1][14][14], int8_t[1][10]);

/*joystick status bits*/
enum {
  JOY_UP = 1u << 0u,
  JOY_DOWN = 1u << 1u,
  JOY_LEFT = 1u << 2u,
  JOY_RIGHT = 1u << 3u,
  JOY_FIRE = 1u << 4u
};

/*C64 interrupt service routines must be handled like ordinary C pointers to a
 * void function*/
typedef void (*ISR_pointer_t)(void);

/*Constants*/
#define CHARSCRXSIZE 40u
#define CHARSCRYSIZE 25u

/*Some important addresses*/
#define ADDR_CURSORX 0xd3u
#define ADDR_CURSORY 0xd6u

#define ADDR_BGCOL 0xd021u
#define ADDR_CURCOL 0x0286u
#define ADDR_FRAMECOL 0xd020u

#define ADDR_COLOR_RAM 0xD800u
#define ADDR_CHAR_RAM 0x0400u

#define ADDR_SPRITE_RAM 0x07F8

#define ACC_CHAR_RAM(x, y) (ADDR(ADDR_CHAR_RAM + (x) + (y)*CHARSCRXSIZE))
#define ACC_COLOR_RAM(x, y) (ADDR(ADDR_COLOR_RAM + (x) + (y)*CHARSCRXSIZE))
#define SPRITE_RAM_SETUP(s) (ADDR(ADDR_SPRITE_RAM + (s)))

/*
volatile uint8_t * const nn_position_indicator_ptr=(volatile
uint8_t*)(0x0400+38+24*40); volatile uint8_t * const
nn_position_indicator2_ptr=(volatile uint8_t*)(0x0400+37+24*40);
*/

/*The rasterizer's interrupt is a nice regular high-frequency event to hook UI
 * things like a status indicator into*/
#define INTERRUPTVEC_RASTERIZER (*(ISR_pointer_t *)0x0314)
#endif