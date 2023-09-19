/* File: scanner.h
 * ---------------
 * You should not need to modify this file. It declare a few constants,
 * types, variables,and functions that are used and/or exported by
 * the lex-generated scanner.
 */

#ifndef _H_scanner
#define _H_scanner

#include <stdio.h>

#define MaxIdentLen 31    // Maximum length for identifiers

extern char *yytext;      // Text of lexeme just scanned

static const char *gTokenNames[27] = {
    "T_Sequential", "T_Conv2d", "T_ReLU", "T_MaxPool2d", "T_AdaptiveAvgPool2d", 
    "T_Linear", "T_Dropout", "T_BatchNorm2d", "T_kernel_size", "T_stride", "T_padding",
    "T_bias", "T_inplace", "T_dilation", "T_ceil_mode", "T_output_size", "T_in_features",
    "T_out_features", "T_p", "T_eps", "T_momentum", "T_affline", "T_track_running_stats", 
    "T_Identifier", "T_IntConstant", "T_DoubleConstant", "T_BoolConstant"
};


int yylex();              // Defined in the generated lex.yy.c file
void yyrestart(FILE *fp); // ditto


void InitScanner();                 // Defined in scanner.l user subroutines
const char *GetLineNumbered(int n); // ditto
 
#endif
