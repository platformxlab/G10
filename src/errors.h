/* File: errors.h
 * --------------
 * This file defines an error-reporting class with a set of already
 * implemented static methods for reporting the standard Decaf errors.
 * You should report all errors via this class so that your error
 * messages will have the same wording/spelling as ours and thus
 * diff can easily compare the two. If needed, you can add new
 * methods if you have some fancy error-reporting, but for the most
 * part, you will just use the class as given.
 */

#ifndef _H_errors
#define _H_errors

#include <string>
using std::string;
#include "location.h"

/* General notes on using this class
 * ----------------------------------
 * Each of the methods in thie class matches one of the standard Decaf
 * errors and reports a specific problem such as an unterminated string,
 * type mismatch, declaration conflict, etc. You will call these methods
 * to report problems encountered during the analysis phases. All methods
 * on this class are static, thus you can invoke methods directly via
 * the class name, e.g.
 *
 *    if (missingEnd) ReportError::UntermString(&yylloc, str);
 *
 * For some methods, the first argument is the pointer to the location
 * structure that identifies where the problem is (usually this is the
 * location of the offending token). You can pass NULL for the argument
 * if there is no appropriate position to point out. For other methods,
 * location is accessed by messaging the node in error which is passed
 * as an argument. You cannot pass NULL for these arguments.
 */


typedef enum {LookingForType, LookingForClass, LookingForInterface, LookingForVariable, LookingForFunction} reasonT;

class ReportError
{
 public:

  // Errors used by preprocessor
  static void UntermComment();
  static void InvalidDirective(int linenum);


  // Errors used by scanner
  static void LongIdentifier(yyltype *loc, const char *ident);
  static void UntermString(yyltype *loc, const char *str);
  static void UnrecogChar(yyltype *loc, char ch);

  // Generic method to report a printf-style error message
  static void Formatted(yyltype *loc, const char *format, ...);


  // Returns number of error messages printed
  static int NumErrors() { return numErrors; }
  
 private:

  static void UnderlineErrorInLine(const char *line, yyltype *pos);
  static void OutputError(yyltype *loc, string msg);
  static int numErrors;
  
};

#endif
