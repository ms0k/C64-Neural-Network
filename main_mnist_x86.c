#include <unistd.h>
#include "misc.h"

const uint8_t number_0_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,237,143,128,213,255,255,255,255,255,255,255,255,255,233,48,2,15,14,219,255,255,255,255,255,255,255,250,94,2,94,226,24,146,255,255,255,255,255,255,255,175,2,54,247,255,128,48,255,255,255,255,255,255,255,69,2,161,255,255,154,59,255,255,255,255,255,255,255,21,12,231,255,255,88,122,255,255,255,255,255,255,255,22,22,255,255,196,3,168,255,255,255,255,255,255,255,22,22,255,222,34,77,243,255,255,255,255,255,255,255,22,10,172,34,36,238,255,255,255,255,255,255,255,255,97,2,2,23,208,255,255,255,255,255,255,255,255,255,246,155,130,240,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

const uint8_t number_1_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,82,234,255,255,255,255,255,255,255,255,255,255,255,255,32,214,255,255,255,255,255,255,255,255,255,255,255,203,11,255,255,255,255,255,255,255,255,255,255,255,255,153,52,255,255,255,255,255,255,255,255,255,255,255,255,143,103,255,255,255,255,255,255,255,255,255,255,255,255,103,123,255,255,255,255,255,255,255,255,255,255,255,255,32,173,255,255,255,255,255,255,255,255,255,255,255,255,1,203,255,255,255,255,255,255,255,255,255,255,255,244,1,244,255,255,255,255,255,255,255,255,255,255,255,255,72,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

const uint8_t number_2_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,251,188,78,30,15,139,255,255,255,255,255,255,255,241,53,36,143,149,212,57,255,255,255,255,255,255,255,251,197,248,255,255,176,46,255,255,255,255,255,255,255,255,255,255,255,255,133,132,255,255,255,255,255,255,255,255,255,255,234,179,21,209,255,255,255,255,255,255,255,255,234,102,4,1,18,160,243,255,255,255,255,255,232,154,116,161,100,77,238,212,247,255,255,255,255,236,60,159,244,107,52,237,255,255,255,255,255,255,255,160,74,157,40,105,237,255,255,255,255,255,255,255,255,138,34,99,191,254,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

const uint8_t number_3_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,250,161,67,30,163,255,255,255,255,255,255,255,255,255,116,75,163,128,21,235,255,255,255,255,255,255,255,255,106,216,255,163,22,255,255,255,255,255,255,255,255,255,252,178,76,1,66,242,255,255,255,255,255,255,255,255,80,1,74,71,5,56,225,255,255,255,255,255,255,255,168,227,255,255,235,37,97,255,255,255,255,255,255,210,255,255,255,255,255,140,66,255,255,255,255,255,255,129,204,255,255,255,238,62,132,255,255,255,255,255,255,174,10,93,146,141,37,46,230,255,255,255,255,255,255,253,151,84,52,67,105,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

const uint8_t number_4_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,244,64,255,255,255,255,255,255,255,255,255,255,255,255,182,107,255,255,255,255,255,255,255,255,255,255,255,255,165,120,255,255,255,255,255,255,255,255,255,255,193,255,165,131,255,255,255,255,255,255,255,255,255,255,31,255,165,120,255,255,255,255,255,255,255,255,255,212,46,215,121,107,255,255,255,255,255,255,255,255,255,154,9,86,64,14,48,216,255,255,255,255,255,255,255,198,200,255,207,53,242,255,255,255,255,255,255,255,255,255,255,255,253,38,255,255,255,255,255,255,255,255,255,255,255,255,255,68,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

const uint8_t number_5_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,151,69,225,255,255,255,255,255,255,255,255,255,255,255,200,109,24,7,91,165,255,255,255,255,255,255,255,255,194,82,249,245,255,255,255,255,255,255,255,255,255,236,31,181,255,255,255,255,255,255,255,255,255,255,255,97,127,255,255,255,255,255,255,255,255,255,255,255,225,3,219,248,255,255,255,255,255,255,255,255,255,255,232,52,1,28,213,255,255,255,255,255,255,255,255,255,255,255,220,62,177,255,255,255,255,255,255,255,255,255,186,252,221,27,231,255,255,255,255,255,255,255,255,255,139,60,28,136,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

const uint8_t number_6_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,251,165,165,255,255,255,255,255,255,255,255,255,255,255,129,2,2,185,255,255,255,255,255,255,255,255,255,215,12,1,8,198,255,255,255,255,255,255,255,255,255,45,2,7,175,255,255,255,255,255,255,255,255,255,198,1,2,84,190,245,255,255,255,255,255,255,255,250,69,2,2,20,2,28,212,255,255,255,255,255,255,226,2,1,2,1,2,4,20,207,255,255,255,255,255,201,2,24,14,2,74,61,2,34,250,255,255,255,255,240,19,11,8,1,5,1,2,40,247,255,255,255,255,255,130,2,2,2,2,7,107,223,255,255,255,255,255,248,245,178,128,128,179,217,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

const uint8_t number_7_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,215,200,200,200,169,128,127,128,223,255,255,255,255,255,46,21,93,93,92,113,92,72,30,190,255,255,255,255,255,255,255,255,255,255,255,255,23,18,255,255,255,255,255,255,255,255,255,255,255,197,2,49,255,255,255,255,255,255,255,255,255,255,231,30,9,154,255,255,255,255,255,255,255,255,255,249,53,2,69,255,255,255,255,255,255,255,255,255,255,167,1,2,218,255,255,255,255,255,255,255,255,255,252,82,2,80,255,255,255,255,255,255,255,255,255,255,202,2,9,226,255,255,255,255,255,255,255,255,255,255,151,2,89,255,255,255,255,255,255,255,255,255,255,255,247,159,239,255,255,255,255,255,255};

const uint8_t number_8_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,244,150,127,127,135,245,255,255,255,255,255,255,255,246,37,27,68,1,1,142,255,255,255,255,255,255,255,117,52,233,174,4,88,237,255,255,255,255,255,255,255,74,48,211,32,129,255,255,255,255,255,255,255,255,255,199,1,1,91,254,255,255,255,255,255,255,255,255,247,117,41,47,44,244,255,255,255,255,255,255,255,255,116,71,245,237,15,138,255,255,255,255,255,255,255,210,4,213,255,255,21,106,255,255,255,255,255,255,255,120,46,255,255,255,21,134,255,255,255,255,255,255,255,160,6,124,173,122,19,199,255,255,255,255,255,255,255,253,170,128,128,137,229,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

const uint8_t number_9_raw[14*14]={255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,141,14,7,70,63,233,255,255,255,255,255,255,255,113,35,205,255,63,0,183,255,255,255,255,255,255,183,28,233,255,247,84,7,205,255,255,255,255,255,255,84,120,247,212,106,7,176,255,255,255,255,255,255,255,127,0,0,0,21,198,255,255,255,255,255,255,255,255,247,183,141,0,198,255,255,255,255,255,255,255,255,255,255,226,21,113,255,255,255,255,255,255,255,255,255,255,247,56,77,247,255,255,255,255,255,255,255,255,255,240,56,21,233,255,255,255,255,255,255,255,255,255,255,183,49,205,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

#include <stdio.h>


extern void entry(const int8_t tensor_Image_Input[1][1][14][14], int8_t tensor_Numeral_OneHot[1][10]);

void select_random_picture(int8_t tens[1][1][14][14]){
	const uint8_t *number_pixaddr[]={number_0_raw, number_1_raw, number_2_raw, number_3_raw, number_4_raw, number_5_raw, number_6_raw, number_7_raw, number_8_raw, number_9_raw};
	int rand_num=rand()%9;
	const uint8_t *number_addr=number_pixaddr[rand_num];
	printf("Placing random number into memory: %d\n", rand_num);
	for(int y=0; y<14; y++){
		for(int x=0; x<14; x++){
			int32_t val=(255u-number_addr[x+y*14])/2u;
			tens[0][0][y][x]=val;
		}
	}
}

void do_mnist(void){
	int8_t tens_input[1][1][14][14];
	int8_t tens_output[1][10];
	select_random_picture(tens_input);
	while(1){
		select_random_picture(tens_input);
		/*for(int y=0; y<14; y++){
			for(int x=0; x<14; x++){
				printf("%03d ", tens_input[0][0][x][y]);
			}
			printf("\n");
		}*/
		entry(tens_input, tens_output);
		int8_t maxresult=-127, maxresultind=-1;
		printf("PROBABILITIES:\n");
		for(uint8_t i=0; i<10; i++){
			printf("%d: %4d\n", i, tens_output[0][i]);
			if(tens_output[0][i]>maxresult){maxresult=tens_output[0][i]; maxresultind=i;}
		}
		printf("MOST PROBABLE NUMBER: %d\n", maxresultind);
		sleep(1.0);
	}
}

int main(void){
	do_mnist();
	return 0;
}