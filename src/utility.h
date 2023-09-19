/* File: utility.h
 * ---------------
 * This file just includes a few support functions you might find
 * helpful in writing the projects (error handling, debug printing)
 */

#ifndef _H_utility
#define _H_utility

#include <stdlib.h>
#include <stdio.h>

/* Function: Failure()
 * Usage: Failure("Out of memory!");
 * --------------------------------
 * Reports an error and exits the program immediately.  You should not
 * need to call this since you should always try to continue parsing,
 * even after an error is encountered.  Some of the provided code calls
 * this in unrecoverable error situations (cannot allocate memory, etc.)
 * Failure accepts printf-style arguments in the message to be printed.
 */
void Failure(const char *format, ...);



/* Macro: Assert()
 * Usage: Assert(num > 0);
 * ----------------------
 * This macro is designed to assert the truth of a necessary condition.
 * It tests the given expression, and if it evalutes true, nothing happens.
 * If it is false, it calls Failure to print a message and abort.
 * For example:  Assert(ptr != NULL)
 * will print something similar to the following if ptr is NULL:
 *   *** Failure: Assertion failed: hashtable.cc, line 55:
 *       ptr != NULL
 */ 
#define Assert(expr)  \
  ((expr) ? (void)0 : Failure("Assertion failed: %s, line %d:\n    %s", __FILE__, __LINE__, #expr))



/* Function: PrintDebug()
 * Usage: PrintDebug("parser", "found ident %s\n", ident);
 * -------------------------------------------------------
 * Print a message if we have turned debugging messages on for the given
 * key.  For example, the usage line shown above will only print a message
 * if the call is preceded by a call to SetDebugForKey("parser",true).
 * The function accepts printf arguments.  The provided main.cc parses
 * the command line to turn on debug flags. 
 */
void PrintDebug(const char *key, const char *format, ...);


/* Function: SetDebugForKey()
 * Usage: SetDebugForKey("scope", true);
 * -------------------------------------
 * Turn on debugging messages for the given key.  See PrintDebug
 * for an example. Can be called manually when desired and will
 * be called from the provided main for flags passed with -d.
 */
void SetDebugForKey(const char *key, bool val);


/* Function: IsDebugOn()
 * Usage: if (IsDebugOn("scope")) ...
 * ----------------------------------
 * Return true/false based on whether this key is currently on
 * for debug printing.
 */
bool IsDebugOn(const char *key);



/* Function: ParseCommandLine
 * --------------------------
 * Turn on the debugging flags from the command line.  Verifies that
 * first argument is -d, and then interpret all the arguments that follow
 * as being flags to turn on.
 */
void ParseCommandLine(int argc, char *argv[]);
     
#endif
