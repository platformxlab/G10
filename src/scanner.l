/* File:  scanner.l
 * ----------------
 * Lex inupt file to generate the scanner for the compiler.
 */
DEC	[0-9]+
DOUBLE	([0-9]+\.[0-9]*([Ee][+-]?[0-9]+)?)|([0-9]([Ee][+-]?[0-9]+)?)
STRING	\"[^\n"]*\"
WRONG_STRING	\"[^\n"]*\n
INDENTIFIER	[a-zA-Z_][a-zA-Z0-9_]*
%{

/* The text within this first region delimited by %{ and %} is assumed to
 * be C/C++ code and will be copied verbatim to the lex.yy.c file ahead
 * of the definitions of the yylex() function. Add other header file inclusions
 * or C++ variable declarations/prototypes that are needed by your code here.
 */

#include <string.h>
#include "scanner.h"
#include "utility.h" // for PrintDebug()
#include "errors.h"
#include "parser.h"
#include <cstdlib>

//static int curLineNum, curColNum;


/* Global variable: yylval
 * -----------------------
 * This global variable is how we get attribute information about the token
 * just scanned to the client. The scanner sets the global variable
 * appropriately and since it's global the client can just read it. this variable will be declared for us in the y.tab.c file
 * produced by Yacc.
 */

/* Macro: YY_USER_ACTION 
 * ---------------------
 * This flex built-in macro can be defined to provide an action which is
 * always executed prior to any matched rule's action. Basically, it is
 * a way of having a piece of code common to all actions factored out to
 * this routine.  We already defined it for you and left the empty
 * function DoBeforeEachAction ready for your use as needed. It will
 * be called once for each pattern scanned from the file, before
 * executing its action.
 */
static void DoBeforeEachAction(); 
#define YY_USER_ACTION DoBeforeEachAction();

%}

 /* The section before the first %% is the Definitions section of the lex
  * input file. Here is where you set options for the scanner, define lex
  * states, and can set up definitions to give names to regular expressions
  * as a simple substitution mechanism that allows for more readable
  * entries in the Rules section later. 
  */


%%             /* BEGIN RULES SECTION */
 /* All patterns and actions should be placed between the start and stop
  * %% markers which delimit the Rules section. 
  */ 
Sequential	{return T_Sequential;}
Conv2d	{return T_Conv2d;}
ReLU	{return T_ReLU;}
MaxPool2d	{return T_MaxPool2d;}
AdaptiveAvgPool2d	{return T_AdaptiveAvgPool2d;}
Linear	{return T_Linear;}
Dropout	{return T_Dropout;}
BatchNorm2d	{return T_BatchNorm2d;}
kernel_size	{return T_kernel_size;}
stride	{return T_stride;}
padding	{return T_padding;}
bias	{return T_bias;}
inplace	{return T_inplace;}
dilation	{return T_dilation;}
ceil_mode	{return T_ceil_mode;}
output_size	{return T_output_size;}
in_features	{return T_in_features;}
out_features	{return T_out_features;}
p	{return T_p;}
eps	{return T_eps;}
momentum	{return T_momentum;}
affine	{return T_affine;}
track_running_stats	{return T_track_running_stats;}
:	{return ':';}
=	{return '=';}
,	{return ',';}
\(	{return '(';}
\)	{return ')';}
True	{yylval.boolConstant = true; return T_BoolConstant;}
False	{yylval.boolConstant = false; return T_BoolConstant;}
{DEC}	{yylval.integerConstant = strtol(yytext, NULL, 10); return T_IntConstant;}
{DOUBLE}	{yylval.doubleConstant = strtod(yytext, NULL); return T_DoubleConstant;}
{INDENTIFIER}	{
					if (strlen(yytext) > MaxIdentLen) {
						ReportError::LongIdentifier(&yylloc, yytext);
					}
					strncpy(yylval.identifier, yytext, MaxIdentLen);
					return T_Identifier;
				}
[ \t]	
\n	{yylloc.last_column = yylloc.first_column = 0; yylineno++;}
.	{ReportError::UnrecogChar(&yylloc, yytext[0]);}
%%
/* The closing %% above marks the end of the Rules section and the beginning
 * of the User Subroutines section. All text from here to the end of the
 * file is copied verbatim to the end of the generated lex.yy.c file.
 * This section is where you put definitions of helper functions.
 */


/* Function: InitScanner
 * ---------------------
 * This function will be called before any calls to yylex().  It is designed
 * to give you an opportunity to do anything that must be done to initialize
 * the scanner (set global variables, configure starting state, etc.). One
 * thing it already does for you is assign the value of the global variable
 * yy_flex_debug that controls whether flex prints debugging information
 * about each token and what rule was matched. If set to false, no information
 * is printed. Setting it to true will give you a running trail that might
 * be helpful when debugging your scanner. Please be sure the variable is
 * set to false when submitting your final version.
 */
void InitScanner()
{
    PrintDebug("lex", "Initializing scanner");
    yy_flex_debug = false;
	yylloc.first_column = yylloc.last_column = 0;
	yylineno = 1;
}


/* Function: DoBeforeEachAction()
 * ------------------------------
 * This function is installed as the YY_USER_ACTION. This is a place
 * to group code common to all actions.
 */
static void DoBeforeEachAction()
{
	yylloc.first_line = yylloc.last_line = yylineno;
	yylloc.first_column = yylloc.last_column + 1;
	yylloc.last_column = yylloc.first_column + yyleng - 1;
}
