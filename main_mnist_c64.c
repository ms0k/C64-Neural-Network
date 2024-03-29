#include "misc.h"

/*example numbers*/
const uint8_t three_raw[14 * 14] = {
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00,
    0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff};

const uint8_t four_raw[14 * 14] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff,
    0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff};

const uint8_t five_raw[14 * 14] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff};
#include <stdio.h>

/*Set cursor position*/
__attribute__((always_inline)) void setcursor(uint8_t x, uint8_t y) {
  const volatile void (*set_cursor)(void) = (const volatile void (*)(
      void))0xe56cu; /*C64 KERNAL "syscall" to set cursor position*/
  ADDR(ADDR_CURSORX) = x;
  ADDR(ADDR_CURSORY) = y;
  set_cursor();
}

void drawborder(uint8_t col) {
  /*Draw border using checkers" pattern character*/
  for (int i = 0; i < 16; i++) {
    ACC_CHAR_RAM(0, i) = 0x66;
    ACC_COLOR_RAM(0, i) = col;
    ACC_CHAR_RAM(15, i) = 0x66;
    ACC_COLOR_RAM(15, i) = col;
    ACC_CHAR_RAM(i, 0) = 0x66;
    ACC_COLOR_RAM(i, 0) = col;
    ACC_CHAR_RAM(i, 15) = 0x66;
    ACC_COLOR_RAM(i, 15) = col;
  }
}

ISR_pointer_t original_raster_interrupt;

volatile uint8_t *const loading_indicator_ptr =
    (volatile uint8_t *)(0x0400 + 39 + 24 * 40);
volatile uint8_t *const loading_indicator_color_ptr =
    (volatile uint8_t *)(0xD800 + 39 + 24 * 40);
const uint8_t working_indicator_chars[] = {0x42, 0x4e, 0x43, 0x4d};
uint8_t currently_working = 0;

/*
This is hooked into the rasterizer ISR to display a "working" indicator. A very
hacky function that might have better been implementer in ASM, since usage of
the stack is not allowed.
*/
__attribute__((interrupt)) __attribute__((no_isr))
__attribute__((noreturn)) static void
timer_interrupt(void) {
  if (currently_working)
    *loading_indicator_ptr = working_indicator_chars[CIA1.tod_10 % 4];
  __asm__("asl $d019");
  __asm__("jmp $ea31");
  exit(0);
}

/*Copy a randomly selected numeral picture into tensor memory and VRAM*/
void load_random_picture(int8_t tens[1][1][14][14]) {
  const uint8_t *number_pixaddr[] = {three_raw, four_raw, five_raw};
  /*Choose random number bitmap from rasterizer position, which is a reasonably
   * good random value especially when running inside an emulator*/
  const uint8_t *number_addr = number_pixaddr[(VIC.rasterline ^ 137) % 3];
  /*Fill the drawing area with the picture (sort of)*/
  for (int y = 0; y < 14; y++) {
    for (int x = 0; x < 14; x++) {
      tens[0][0][y][x] = (255u - number_addr[x + y * 14]) / 2u;
      ACC_CHAR_RAM(x + 1, y + 1) = tens[0][0][y][x] > 16 ? 224 : 96;
      ACC_COLOR_RAM(x + 1, y + 1) = 1;
    }
  }
  drawborder(COLOR_WHITE);
}

/*Return time of day from interface chips*/
__attribute__((always_inline)) void readtimeofday(uint8_t time[]) {
  time[0] = CIA1.tod_hour;
  time[1] = CIA1.tod_min;
  time[2] = CIA1.tod_sec;
  time[3] = CIA1.tod_10;
  time[0]--;
}

/*Convert a BCD number to a "true" binary number*/
__attribute__((always_inline)) void convert_bcd(uint8_t time[]) {
  for (int i = 0; i < 4; i++) {
    time[i] = (time[i] & 0xfu) + ((time[i] >> 4u) & 0xfu) * 10;
  }
}

/*Preferring this over OS syscalls*/
void busywait(volatile uint8_t v1, volatile uint8_t v2) {
  for (volatile int i = 0; i < v1; i++)
    for (volatile int j = 0; j < v2; j++) {
    }
}

uint8_t colorpalette[16] = {
    COLOR_BLACK,      COLOR_GRAY1,  COLOR_BROWN,     COLOR_RED,
    COLOR_PURPLE,     COLOR_BLUE,   COLOR_ORANGE,    COLOR_GRAY2,
    COLOR_LIGHTRED,   COLOR_GREEN,  COLOR_LIGHTBLUE, COLOR_GRAY3,
    COLOR_LIGHTGREEN, COLOR_YELLOW, COLOR_CYAN,      COLOR_WHITE};

/*
Dithered bitmap mode would be nicer, but not sure how to allocate large memory
regions with C programs. Unused functional because it requires changing the
generated C code, but tested as working. Displays [dim1] feature maps of size
[dim2*dim3] from a tensor array.
*/
void display_featuremaps(int8_t *featuremaps, uint8_t dim1, uint8_t dim2,
                         uint8_t dim3) {
  memset((void *)ADDR_CHAR_RAM, 224, 25u * 40u);
  memset((void *)ADDR_COLOR_RAM, COLOR_BLACK, 25u * 40u);
  for (uint8_t i = 0; i < 16; i++)
    ACC_COLOR_RAM(i, 24) = colorpalette[i];
  setcursor(0, 22);
  printf("REFERENCE COLOR SCALE");
  setcursor(0, 23);
  printf("0              255");
  *loading_indicator_color_ptr = COLOR_WHITE;
  uint8_t xslot = 0, yslot = 0;
  for (uint8_t m = 0; m < dim1; m++) {
    uint8_t *map = (uint8_t *)&featuremaps[m * dim2 * dim3];
    for (uint8_t y = 0; y < dim3; y++) {
      for (uint8_t x = 0; x < dim2; x++) {
        ACC_COLOR_RAM(xslot + x, yslot + y) =
            colorpalette[map[x + y * dim3] / 16];
      }
    }
    xslot += dim2 + 1;
    if (xslot >= 40) {
      xslot = 0;
      yslot += dim3 + 1;
      if (yslot >= 25) {
        yslot = 0;
        busywait(100, 100);
      }
    }
  }
}

const uint8_t cursprite[] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x81, 0x00, 0x00,
    0x42, 0x00, 0x00, 0x24, 0x00, 0x00, 0x18, 0x00, 0x00, 0x18, 0x00,
    0x00, 0x24, 0x00, 0x00, 0x42, 0x00, 0x00, 0x81, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03};

/*Function to get a painted number from the user*/
void do_paint(int8_t tens_input[1][1][14][14]) {
  uint8_t pointpos[2] = {7, 7};
  uint8_t pixel_flipped = 0;
  /*Setup cursor sprite*/
  for (int i = 0; i < 64; i++)
    ADDR(0x2c0 + i) = cursprite[i];
  VIC.spr_color[0] = COLOR_GRAY3;
  VIC.spr_ena |= 1u;
  SPRITE_RAM_SETUP(0) = 11;
  while (ADDR(0xcb) == 1) {
  } /*Wait until "enter" released*/
  while (1) {
    if (ADDR(0xc5) == 1)
      break; /*Key pressed "enter"*/
    /*Move on joystick axis or arrow keypress*/
    pointpos[1] -=
        ((ADDR(0xcb) == 7 && (ADDR(0x28d) & 1)) || !(CIA1.pra & JOY_UP)) &&
        pointpos[1] > 1;
    pointpos[1] +=
        ((ADDR(0xcb) == 7 && !(ADDR(0x28d) & 1)) || !(CIA1.pra & JOY_DOWN)) &&
        pointpos[1] < 14;
    pointpos[0] -=
        ((ADDR(0xcb) == 2 && (ADDR(0x28d) & 1)) || !(CIA1.pra & JOY_LEFT)) &&
        pointpos[0] > 1;
    pointpos[0] +=
        ((ADDR(0xcb) == 2 && !(ADDR(0x28d) & 1)) || !(CIA1.pra & JOY_RIGHT)) &&
        pointpos[0] < 14;
    if (((~CIA1.pra) & (JOY_UP | JOY_DOWN | JOY_LEFT | JOY_RIGHT)) ||
        ADDR(0xcb) == 2 || ADDR(0xcb) == 7)
      pixel_flipped = 0;
    /*Spacebar or joystick fire*/
    if ((ADDR(0xcb) == 60 || !(CIA1.pra & JOY_FIRE))) {
      if (!pixel_flipped) {
        uint8_t tensorpos[2] = {pointpos[0] - 1, pointpos[1] - 1};
        if (tens_input[0][0][tensorpos[1]][tensorpos[0]] > 16) {
          tens_input[0][0][tensorpos[1]][tensorpos[0]] = 0;
        } else {
          tens_input[0][0][tensorpos[1]][tensorpos[0]] = 127u;
        }
        ACC_CHAR_RAM(pointpos[0], pointpos[1]) =
            tens_input[0][0][tensorpos[1]][tensorpos[0]] > 16 ? 224 : ' ';
        pixel_flipped = 1;
      }
    } else {
      pixel_flipped = 0;
    }
    /*Key pressed "c" -> clear the field*/
    if (ADDR(0xc5) == 20) {
      for (uint8_t y = 0; y < 14; y++) {
        for (uint8_t x = 0; x < 14; x++) {
          tens_input[0][0][y][x] = 0;
          ACC_CHAR_RAM(x + 1, y + 1) = ' ';
        }
      }
    }
    /*Flip picture pixels*/
    if (pixel_flipped) {
      VIC.spr_color[0] = COLOR_YELLOW;
    } else {
      VIC.spr_color[0] =
          VIC.spr_color[0] == COLOR_GRAY3 ? COLOR_GRAY1 : COLOR_GRAY3;
    }
    /*Display the time*/
    VIC.spr_pos[0].x = pointpos[0] * 8 + 24 - 8;
    VIC.spr_pos[0].y = pointpos[1] * 8 + 50 - 6;
    ACC_COLOR_RAM(pointpos[0], pointpos[1]) = COLOR_RED;
    busywait(30, 20);
    ACC_COLOR_RAM(pointpos[0], pointpos[1]) = COLOR_WHITE;
    uint8_t t[4];
    readtimeofday(t);
    convert_bcd(t);
    setcursor(0, 24);
    printf("%02d:%02d:%02d", t[0], t[1], t[2]);
  }
  VIC.spr_ena &= ~1u;
}

void do_mnist(void) {
  /*Tensor input and output buffers, start with random picture*/
  int8_t tens_input[1][1][14][14];
  int8_t tens_output[1][10];
  load_random_picture(tens_input);
  while (1) {
    /*Wait for user to process and confirm the picture, then run recognition*/
    do_paint(tens_input);
    uint8_t time1[4], time2[4];
    readtimeofday(time1);
    drawborder(COLOR_RED);
    currently_working = 1;
    entry(tens_input, tens_output);
    readtimeofday(time2);
    currently_working = 0;
    drawborder(COLOR_WHITE);
    convert_bcd(time1);
    convert_bcd(time2);
    /*Redraw picture*/
    memset((void *)ADDR_CHAR_RAM, ' ', 25u * 40u);
    memset((void *)ADDR_COLOR_RAM, COLOR_WHITE, 25u * 40u);
    for (int y = 0; y < 14; y++) {
      for (int x = 0; x < 14; x++) {
        ACC_CHAR_RAM(x + 1, y + 1) = tens_input[0][0][y][x] > 16 ? 224 : 96;
      }
    }
    /*Display results and calculation duration*/
    drawborder(COLOR_WHITE);
    int8_t maxresult = -127, maxresultind = -1;
    for (uint8_t i = 0; i < 10; i++) {
      if (tens_output[0][i] > maxresult) {
        maxresult = tens_output[0][i];
        maxresultind = i;
      }
    }
    setcursor(17, 0);
    printf("OUTPUTS:");
    for (uint8_t i = 0; i < 10; i++) {
      setcursor(17, i + 1);
      printf("%d: %4d", i, tens_output[0][i]);
      for (uint8_t j = 0; j < 7; j++) {
        ACC_COLOR_RAM(17 + j, i + 1) =
            (i == maxresultind) ? COLOR_YELLOW : COLOR_WHITE;
      }
    }
    int16_t total_time = (time2[0] - time1[0]) * 60 * 60 +
                         (time2[1] - time1[1]) * 60 + (time2[2] - time1[2]);
    setcursor(0, 17);
    printf("%02d:%02d:%02d.%d->%02d:%02d:%02d.%d->%4dS RUNTIME", time1[0],
           time1[1], time1[2], time1[3], time2[0], time2[1], time2[2], time2[3],
           total_time);
    setcursor(0, 19);
    printf("MOST PROBABLE NUMBER: %d", maxresultind);
  }
}

int main(void) {
  /*Set CIA1 timer for approximate benchmarking*/
  CIA1.tod_10 = 0;
  /*Insert rasterizer ISR*/
  if (1) {
    SEI();
    original_raster_interrupt = INTERRUPTVEC_RASTERIZER;
    INTERRUPTVEC_RASTERIZER = timer_interrupt;
    CLI();
  }
  /*Set background, font and frame color*/
  VIC.bgcolor0 = COLOR_BLACK;
  ADDR(ADDR_CURCOL) = COLOR_WHITE;
  VIC.bordercolor = COLOR_GRAY2;
  for (int y = 0; y < 25; y++) {
    for (int x = 0; x < 40; x++) {
      ACC_CHAR_RAM(x, y) = 96;
      ACC_COLOR_RAM(x, y) = 1;
    }
  }
  /*Run the MNIST demo*/
  do_mnist();
  return 0;
}