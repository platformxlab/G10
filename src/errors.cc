/* File: errors.cc
 * ---------------
 * Implementation for error-reporting class.
 */

#include "errors.h"
#include <iostream>
#include <sstream>
#include <stdarg.h>
#include <stdio.h>
using namespace std;



int ReportError::numErrors = 0;

 
void ReportError::OutputError(yyltype *loc, string msg) {
    numErrors++;
    fflush(stdout); // make sure any buffered text has been output
    if (loc) {
        cerr << endl << "*** Error line " << loc->first_line << "." << endl;
    } else
        cerr << endl << "*** Error." << endl;
    cerr << "*** " << msg << endl << endl;
}


void ReportError::Formatted(yyltype *loc, const char *format, ...) {
    va_list args;
    char errbuf[2048];
    
    va_start(args, format);
    vsprintf(errbuf,format, args);
    va_end(args);
    OutputError(loc, errbuf);
}

void ReportError::UntermComment() {
    OutputError(NULL, "Input ends with unterminated comment");
}

void ReportError::InvalidDirective(int linenum) {
    yyltype ll = {0, linenum, 0, 0};
    OutputError(&ll, "Invalid # directive");
}

void ReportError::LongIdentifier(yyltype *loc, const char *ident) {
    ostringstream s;
    s << "Identifier too long: \"" << ident << "\"";
    OutputError(loc, s.str());
}

void ReportError::UntermString(yyltype *loc, const char *str) {
    ostringstream s;
    s << "Unterminated string constant: " << str;
    OutputError(loc, s.str());
}

void ReportError::UnrecogChar(yyltype *loc, char ch) {
    ostringstream s;
    s << "Unrecognized char: '" << ch << "'";
    OutputError(loc, s.str());
}

/* Function: yyerror()
 * -------------------
 * Standard error-reporting function expected by yacc. Our version merely
 * just calls into the error reporter above, passing the location of
 * the last token read. If you want to suppress the ordinary "parse error"
 * message from yacc, you can implement yyerror to do nothing and
 * then call ReportError::Formatted yourself with a more descriptive 
 * message.
 */
void yyerror(const char *msg) {
    ReportError::Formatted(&yylloc, "%s", msg);
}
