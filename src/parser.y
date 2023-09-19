

%{

/* Just like lex, the text within this first region delimited by %{ and %}
 * is assumed to be C/C++ code and will be copied verbatim to the y.tab.c
 * file ahead of the definitions of the yyparse() function. Add other header
 * file inclusions or C++ variable declarations/prototypes that are needed
 * by your code here.
 */
#define YYDEBUG 1
#include "scanner.h" // for yylex
#include "parser.h"
#include "errors.h"

void yyerror(const char *msg); // standard error-handling routine


%}

/* The section before the first %% is the Definitions section of the yacc
 * input file. Here is where you declare tokens and types, add precedence
 * and associativity options, and so on.
 */
 
/* yylval 
 * ------
 * Here we define the type of the yylval global variable that is used by
 * the scanner to store attibute information about the token just scanned
 * and thus communicate that information to the parser. 
 *
 */
%union {
    int integerConstant;
    bool boolConstant;
    char *stringConstant;
    double doubleConstant;
    char identifier[MaxIdentLen+1]; // +1 for terminating null
    Layer *layer;
    List<Layer*> *layerlist;
    Block *block;
    SeqBlock *seqblock;
    List<SeqBlock*> *seqblocklist;
    Operatorr *operatorr;
}


/* Tokens
 * ------
 * Here we tell yacc about all the token types that we are using.
 * Yacc will assign unique numbers to these and export the #define
 * in the generated y.tab.h header file.
 */

%token   T_Sequential T_Conv2d T_ReLU T_MaxPool2d T_AdaptiveAvgPool2d
%token   T_Linear T_Dropout T_BatchNorm2d T_kernel_size T_stride T_padding
%token   T_bias T_inplace T_dilation T_ceil_mode T_output_size T_in_features
%token   T_out_features T_p T_eps T_momentum T_affine T_track_running_stats


%token   <identifier> T_Identifier
%token   <integerConstant> T_IntConstant
%token   <doubleConstant> T_DoubleConstant
%token   <boolConstant> T_BoolConstant


/* Non-terminal types
 * ------------------
 * In order for yacc to assign/access the correct field of $$, $1, we
 * must to declare which field is appropriate for the non-terminal.
 * As an example, this first type declaration establishes that the DeclList
 * non-terminal uses the field named "declList" in the yylval union. This
 * means that when we are setting $$ for a reduction for DeclList ore reading
 * $n which corresponds to a DeclList nonterminal we are accessing the field
 * of the union named "declList" which is of type List<Decl*>.
 * pp2: You'll need to add many of these of your own.
 */
%type <layerlist>   LayerList
%type <layer>   Layer BlockLayer SeqBlockLayer
%type <block>   Block UserBlock OperatorBlock
%type <seqblocklist>    SeqBlockList
%type <seqblock>    SeqBlock
%type <operatorr>    Operatorr //Conv2d ReLU MaxPool2d AdaptiveAvgPool2d Linear Dropout BatchNorm2d


%nonassoc '='
%nonassoc ','
%nonassoc '(' ')'
%left ':'

%start Model
%%
/* Rules
 * -----
 * All productions and actions should be placed between the start and stop
 * %% markers which delimit the Rules section.
	 
 */
Model       :   T_Identifier '(' LayerList ')'         { 
                                                        @1; 
                                                        /* pp2: The @1 is needed to convince 
                                                        * yacc to set up yylloc. You can remove 
                                                        * it once you have other uses of @n*/
                                                        Identifier *id = new Identifier(@1, $1);
                                                        Model *model = new Model(id, $3);
                                                        // if(ReportError::NumErrors() == 0)
                                                        //     model->Print(0);
                                                        model->Analysis();
                                                        }
            ;

LayerList   :   LayerList Layer     { ($$=$1)->Append($2); }
            |   Layer               { ($$ = new List<Layer*>)->Append($1); }
            ;

Layer       :   BlockLayer           { $$ = $1;}
            |   SeqBlockLayer        { $$ = $1;}
            ;

BlockLayer  :   '(' T_Identifier ')' ':' Block      { 
                                                        Identifier* id = new Identifier(@2, $2);
                                                        BlockLayer* bl = new BlockLayer(id, $5);
                                                        $$ = bl;
                                                    }
            ;

SeqBlockLayer:  '(' T_Identifier ')' ':' T_Sequential '(' SeqBlockList ')'  
                { 
                    Identifier* id = new Identifier(@2, $2);
                    SeqBlockLayer* sbl = new SeqBlockLayer(id, $7);
                    $$ = sbl;
                }
            ;

SeqBlockList:   SeqBlockList SeqBlock   { ($$=$1)->Append($2); }
            |   SeqBlock                { ($$ = new List<SeqBlock*>)->Append($1); }
            ;

Block       :   UserBlock               { $$ = $1;}
            |   OperatorBlock           { $$ = $1;}
            ;

UserBlock   :   T_Identifier '(' LayerList ')'
                {   Identifier* id = new Identifier(@1, $1);
                    UserBlock* ub = new UserBlock(id, $3);
                    $$ = ub;
                }
            ;

OperatorBlock:  Operatorr            { $$ = new OperatorBlock($1);}
            ;

SeqBlock    :   '(' T_IntConstant ')' ':' Block     {
                                                        SeqBlock* sbk = new SeqBlock($2, $5);
                                                        $$ = sbk;
                                                    }
            |   '(' T_Identifier ')' ':' Block      {
                                                        SeqBlock* sbk = new SeqBlock(-1, $5);
                                                        $$ = sbk;
                                                    }
            ;

Operatorr   :   T_Conv2d '(' T_IntConstant ',' T_IntConstant ',' T_kernel_size '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_stride '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_padding '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_bias '=' T_BoolConstant ')' 
                {
                    
                    $$ = new Conv2d(@1, $3, $5, $10, $12, $18, $20, $26, $28, $33);
                }
            |   T_Conv2d '(' T_IntConstant ',' T_IntConstant ',' T_kernel_size '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_stride '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_bias '=' T_BoolConstant ')'
                {
                    Conv2d* op = new Conv2d(@1, $3, $5, $10, $12, $18, $20, 0, 0, $25);
                    $$ = op;
                }
            |   T_Conv2d '(' T_IntConstant ',' T_IntConstant ',' T_kernel_size '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_stride '=' '(' T_IntConstant ',' T_IntConstant ')' ',' T_padding '=' '(' T_IntConstant ',' T_IntConstant ')' ')'
                {
                    Conv2d* op = new Conv2d(@1, $3, $5, $10, $12, $18, $20, $26, $28);
                    $$ = op;
                }
            |   T_ReLU '(' T_inplace '=' T_BoolConstant ')' 
                {
                    ReLU* op = new ReLU(@1, $5);
                    $$ = op;
                }
            |   T_MaxPool2d '(' T_kernel_size '=' T_IntConstant ',' T_stride '=' T_IntConstant ',' T_padding '=' T_IntConstant ',' T_dilation '=' T_IntConstant ',' T_ceil_mode '=' T_BoolConstant ')'
                {
                    MaxPool2d* op = new MaxPool2d(@1, $5, $9, $13, $17, $21);
                    $$ = op;
                }
            |   T_AdaptiveAvgPool2d '(' T_output_size '=' '(' T_IntConstant ',' T_IntConstant ')' ')'
                {
                    AdaptiveAvgPool2d* op = new AdaptiveAvgPool2d(@1, $6, $8);
                    $$ = op;
                }
            |   T_Linear '(' T_in_features '=' T_IntConstant ',' T_out_features '=' T_IntConstant ',' T_bias '=' T_BoolConstant ')'
                {
                    Linear* op = new Linear(@1, $5, $9, $13);
                    $$ = op;
                }
            |   T_Dropout '(' T_p '=' T_DoubleConstant ',' T_inplace '=' T_BoolConstant ')'
                {
                    Dropout* op = new Dropout(@1, $5, $9);
                    $$ = op;
                }
            |   T_BatchNorm2d '(' T_IntConstant ',' T_eps '=' T_DoubleConstant ',' T_momentum '=' T_DoubleConstant ',' T_affine '=' T_BoolConstant ',' T_track_running_stats '=' T_BoolConstant ')'
                {
                    BatchNorm2d* op = new BatchNorm2d(@1, $3, $7, $11, $15, $19);
                    $$ = op;
                }
            ;

%%

/* The closing %% above marks the end of the Rules section and the beginning
 * of the User Subroutines section. All text from here to the end of the
 * file is copied verbatim to the end of the generated y.tab.c file.
 * This section is where you put definitions of helper functions.
 */

/* Function: InitParser
 * --------------------
 * This function will be called before any calls to yyparse().  It is designed
 * to give you an opportunity to do anything that must be done to initialize
 * the parser (set global variables, configure starting state, etc.). One
 * thing it already does for you is assign the value of the global variable
 * yydebug that controls whether yacc prints debugging information about
 * parser actions (shift/reduce) and contents of state stack during parser.
 * If set to false, no information is printed. Setting it to true will give
 * you a running trail that might be helpful when debugging your parser.
 * Please be sure the variable is set to false when submitting your final
 * version.
 */
void InitParser()
{
   PrintDebug("parser", "Initializing parser");
   yydebug = false;
}

