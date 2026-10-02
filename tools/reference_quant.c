// Independent oracle: linked to untouched llama.cpp b11284 GGML dequantizers.
#include "ggml-quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
int main(int argc,char** argv){
    if(argc!=5){fprintf(stderr,"usage: reference-quant TYPE ELEMENTS INPUT OUTPUT\n");return 2;}
    int type=atoi(argv[1]);int64_t n=atoll(argv[2]);if(n<=0||n>16000000||n%256){return 2;}
    FILE* f=fopen(argv[3],"rb");if(!f)return 3;fseek(f,0,SEEK_END);long len=ftell(f);rewind(f);
    if(len<=0||len>128*1024*1024){fclose(f);return 3;}void* raw=malloc(len);float* y=malloc(n*sizeof(float));if(!raw||!y)return 3;
    if(fread(raw,1,len,f)!=(size_t)len)return 3;fclose(f);
    switch(type){
        case 0:if(len!=n*4)return 4;for(int64_t i=0;i<n;i++)y[i]=((float*)raw)[i];break;
        case 8:if(len!=n/32*34)return 4;dequantize_row_q8_0(raw,y,n);break;
        case 10:if(len!=n/256*84)return 4;dequantize_row_q2_K(raw,y,n);break;
        case 11:if(len!=n/256*110)return 4;dequantize_row_q3_K(raw,y,n);break;
        case 12:if(len!=n/256*144)return 4;dequantize_row_q4_K(raw,y,n);break;
        case 13:if(len!=n/256*176)return 4;dequantize_row_q5_K(raw,y,n);break;
        case 14:if(len!=n/256*210)return 4;dequantize_row_q6_K(raw,y,n);break;
        case 17:if(len!=n/256*74)return 4;dequantize_row_iq2_xs(raw,y,n);break;
        case 18:if(len!=n/256*98)return 4;dequantize_row_iq3_xxs(raw,y,n);break;
        case 20:if(len!=n/32*18)return 4;dequantize_row_iq4_nl(raw,y,n);break;
        case 21:if(len!=n/256*110)return 4;dequantize_row_iq3_s(raw,y,n);break;
        case 22:if(len!=n/256*82)return 4;dequantize_row_iq2_s(raw,y,n);break;
        case 23:if(len!=n/256*136)return 4;dequantize_row_iq4_xs(raw,y,n);break;
        default:fprintf(stderr,"unsupported oracle type\n");return 4;
    }
    f=fopen(argv[4],"wb");if(!f)return 5;size_t written=fwrite(y,sizeof(float),n,f);fclose(f);free(raw);free(y);return written==(size_t)n?0:5;
}
