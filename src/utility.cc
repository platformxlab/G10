/* File: utiliy.cc
 * ---------------
 * Implementation of simple printing functions to report failures or
 * debugging information triggered by keys.
 */

#include "utility.h"
#include <stdarg.h>
#include <string.h>
#include "list.h"

static List<const char*> debugKeys;
static const int BufferSize = 2048;

void Failure(const char *format, ...)
{
  va_list args;
  char errbuf[BufferSize];
  
  va_start(args, format);
  vsprintf(errbuf, format, args);
  va_end(args);
  fflush(stdout);
  fprintf(stderr,"\n*** Failure: %s\n\n", errbuf);
  abort();
}



int IndexOf(const char *key)
{
   for (int i = 0; i < debugKeys.NumElements(); i++)
      if (!strcmp(debugKeys.Nth(i), key)) return i;
   return -1;
}

bool IsDebugOn(const char *key)
{
   return (IndexOf(key) != -1);
}


void SetDebugForKey(const char *key, bool value)
{
  int k = IndexOf(key);
  if (!value && k != -1)
    debugKeys.RemoveAt(k);
  else if (value && k == -1)
    debugKeys.Append(key);
}



void PrintDebug(const char *key, const char *format, ...)
{
  va_list args;
  char buf[BufferSize];

  if (!IsDebugOn(key))
     return;
  
  va_start(args, format);
  vsprintf(buf, format, args);
  va_end(args);
  printf("+++ (%s): %s%s", key, buf, buf[strlen(buf)-1] != '\n'? "\n" : "");
}


void ParseCommandLine(int argc, char *argv[])
{
  if (argc <= 2)
    return;
  
  if (strcmp(argv[2], "-d") != 0) { // first arg is not -d
    printf("Usage:   -d <debug-key-1> <debug-key-2> ... \n");
    exit(2);
  }

  for (int i = 3; i < argc; i++)
    SetDebugForKey(argv[i], true);
}

