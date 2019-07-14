/*************************************************************************/
/*									 */
/*  Copyright 2010 Rulequest Research Pty Ltd.				 */
/*									 */
/*  This file is part of GritBot GPL Edition, a single-threaded version	 */
/*  of GritBot release 2.01.						 */
/*									 */
/*  GritBot GPL Edition is free software: you can redistribute it	 */
/*  and/or modify it under the terms of the GNU General Public License	 */
/*  as published by the Free Software Foundation, either version 3 of	 */
/*  the License, or (at your option) any later version.			 */
/*									 */
/*  GritBot GPL Edition is distributed in the hope that it will be	 */
/*  useful, but WITHOUT ANY WARRANTY; without even the implied warranty	 */
/*  of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	 */
/*  GNU General Public License for more details.			 */
/*									 */
/*  You should have received a copy of the GNU General Public License	 */
/*  (gpl.txt) along with GritBot GPL Edition.  If not, see		 */
/*									 */
/*      <http://www.gnu.org/licenses/>.					 */
/*									 */
/*************************************************************************/



/*************************************************************************/
/*									 */
/*	Get names of classes, attributes and attribute values		 */
/*	-----------------------------------------------------		 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"

#include <sys/types.h>
#include <sys/stat.h>

#define	MAXLINEBUFFER	10000
int	Delimiter;
char	LineBuffer[MAXLINEBUFFER], *LBp=LineBuffer;



/*************************************************************************/
/*									 */
/*	Read a name from file f into string s, setting Delimiter.	 */
/*									 */
/*	- Embedded periods are permitted, but periods followed by space	 */
/*	  characters act as delimiters.					 */
/*	- Embedded spaces are permitted, but multiple spaces are	 */
/*	  replaced by a single space.					 */
/*	- Any character can be escaped by '\'.				 */
/*	- The remainder of a line following '|' is ignored.		 */
/*									 */
/*	Colons are sometimes delimiters depending on ColonOpt		 */
/*									 */
/*************************************************************************/


Boolean ReadName(FILE *f, String s, int n, char ColonOpt)
/*      --------  */
{
    register char *Sp=s;
    register int  c;
    char	  Msg[2];

    /*  Skip to first non-space character  */

    while ( (c = InChar(f)) == '|' || Space(c) )
    {
	if ( c == '|' ) SkipComment;
    }

    /*  Return false if no names to read  */

    if ( c == EOF )
    {
	Delimiter = EOF;
	return false;
    }

    /*  Read in characters up to the next delimiter  */

    while ( c != ColonOpt && c != ',' && c != '\n' && c != '|' && c != EOF )
    {
	if ( --n <= 0 )
	{
	    if ( Of ) Error(LONGNAME, "", "");
	}

	if ( c == '.' )
	{
	    if ( (c = InChar(f)) == '|' || Space(c) || c == EOF ) break;
	    *Sp++ = '.';
	    continue;
	}

	if ( c == '\\' )
	{
	    c = InChar(f);
	}

	if ( Space(c) )
	{
	    *Sp++ = ' ';

	    while ( ( c = InChar(f) ) == ' ' || c == '\t' )
		;
	}
	else
	{
	    *Sp++ = c;
	    c = InChar(f);
	}
    }

    if ( c == '|' ) SkipComment;
    Delimiter = c;

    /*  Special case for ':='  */

    if ( Delimiter == ':' )
    {
	if ( *LBp == '=' )
	{
	    Delimiter = '=';
	    LBp++;
	}
    }

    /*  Strip trailing spaces  */

    while ( Sp > s && Space(*(Sp-1)) ) Sp--;

    if ( Sp == s )
    {
	Msg[0] = ( Space(c) ? '.' : c );
	Msg[1] = '\00';
	Error(MISSNAME, Fn, Msg);
    }

    *Sp++ = '\0';
    return true;
}



#ifndef INSPECT
/*************************************************************************/
/*									 */
/*	Read names of classes, attributes and legal attribute values.	 */
/*	On completion, names are stored in:				 */
/*	  AttName	-	attribute names				 */
/*	  AttValName	-	attribute value names			 */
/*	with:								 */
/*	  MaxAttVal	-	number of values for each attribute	 */
/*									 */
/*	Other global variables set are:					 */
/*	  MaxAtt	-	maximum attribute number		 */
/*	  MaxDiscrVal	-	maximum discrete values for an attribute */
/*									 */
/*************************************************************************/


void GetNames(FILE *Nf)
/*   --------  */
{
    char	Buffer[1000]="";
    int		AttCeiling=100, ClassCeiling=100;
    DiscrValue	MaxClass=0, v;
    String	*ExplicitClass;
    Attribute	Att;

    ErrMsgs = AttExIn = 0;
    LineNo  = 0;

    /*  Get class names from names file  */

    ExplicitClass = AllocZero(ClassCeiling, String);
    ClassAtt = LabelAtt = 0;
    do
    {
	ReadName(Nf, Buffer, 1000, ':');

	if ( ++MaxClass >= ClassCeiling)
	{
	    ClassCeiling += 100;
	    Realloc(ExplicitClass, ClassCeiling, String);
	}
	ExplicitClass[MaxClass] = strdup(Buffer);
    }
    while ( Delimiter == ',' );

    /*  Ignore thresholds for See5/C5.0 continuous class attribute  */

    if ( Delimiter == ':' )
    {
	do
	{
	    ReadName(Nf, Buffer, 1000, ':');
	}
	while ( Delimiter == ',' );
    }

    /*  Get attribute and attribute value names from names file  */

    AttName = AllocZero(AttCeiling, String);
    MaxAttVal = AllocZero(AttCeiling, DiscrValue);
    AttValName = AllocZero(AttCeiling, String *);
    SpecialStatus = AllocZero(AttCeiling, char);
    AttDef = AllocZero(AttCeiling, Definition);

    MaxAtt = 0;
    while ( ReadName(Nf, Buffer, 1000, ':') )
    {
	if ( Delimiter != ':' && Delimiter != '=' )
	{
	    Error(BADATTNAME, Buffer, "");
	}

	/*  Check for include/exclude instruction  */

	if ( ( *Buffer == 'a' || *Buffer == 'A' ) &&
	     ! memcmp(Buffer+1, "ttributes ", 10) &&
	     ! memcmp(Buffer+strlen(Buffer)-6, "cluded", 6) )
	{
	    AttExIn = ( ! memcmp(Buffer+strlen(Buffer)-8, "in", 2) ? 1 : -1 );
	    if ( AttExIn == 1 )
	    {
		ForEach(Att, 1, MaxAtt)
		{
		    SpecialStatus[Att] |= SKIP;
		}
	    }

	    while ( ReadName(Nf, Buffer, 1000, ':') )
	    {
		Att = Which(Buffer, AttName, 1, MaxAtt);
		if ( ! Att )
		{
		    Error(UNKNOWNATT, Buffer, Nil);
		}
		else
		if ( AttExIn == 1 )
		{
		    SpecialStatus[Att] -= SKIP;
		}
		else
		{
		    SpecialStatus[Att] |= SKIP;
		}
	    }

	    break;
	}

	if ( Which(Buffer, AttName, 1, MaxAtt) > 0 )
	{
	    Error(DUPATTNAME, Buffer, Nil);
	}

	if ( ++MaxAtt >= AttCeiling-1 )		/* ensure space for class att */
	{
	    AttCeiling += 100;
	    Realloc(AttName, AttCeiling, String);
	    Realloc(MaxAttVal, AttCeiling, DiscrValue);
	    Realloc(AttValName, AttCeiling, String *);
	    Realloc(SpecialStatus, AttCeiling, char);
	    Realloc(AttDef, AttCeiling, Definition);
	}

	AttName[MaxAtt] = strdup(Buffer);
	SpecialStatus[MaxAtt] = Nil;
	AttDef[MaxAtt] = Nil;
	MaxAttVal[MaxAtt] = 0;

	if ( Delimiter == '=' )
	{
	    if ( MaxClass == 1 && ! strcmp(ExplicitClass[1], AttName[MaxAtt]) )
	    {
		Error(BADDEF3, Nil, Nil);
	    }

	    ImplicitAtt(Nf);
	}
	else
	{
	    ExplicitAtt(Nf);
	}
    }

    /*  Check whether class is one of the attributes  */

    if ( MaxClass == 1 )
    {
	ClassAtt = Which(ExplicitClass[1], AttName, 1, MaxAtt);
	Free(ExplicitClass[1]);
	Free(ExplicitClass);
    }
    else
    {
	MaxAtt++;
	AttName[MaxAtt] = strdup("class");

	/*  Set up last attribute with values "N/A" and explicit classes  */

	AttValName[MaxAtt] = Alloc(MaxClass+2, String);
	AttValName[MaxAtt][1] = strdup("N/A");
	ForEach(v, 1, MaxClass)
	{
	    AttValName[MaxAtt][v+1] = ExplicitClass[v];
	}
	Free(ExplicitClass);

	MaxAttVal[MaxAtt]  = MaxClass+1;
	MaxDiscrVal	   = Max(MaxDiscrVal, MaxClass+1);

	AttDef[MaxAtt] = Nil;
	SpecialStatus[MaxAtt] = ( AttExIn == 1 ? SKIP : 0 );
    }

    fclose(Nf);

    if ( ErrMsgs > 0 ) Goodbye(1);
}



/*************************************************************************/
/*									 */
/*	Continuous or discrete attribute				 */
/*									 */
/*************************************************************************/


void ExplicitAtt(FILE *Nf)
/*   -----------  */
{
    char	Buffer[1000]="", *p;
    DiscrValue	v;
    int		ValCeiling=100, BaseYear;
    time_t	clock;

    /*  Read attribute type or first discrete value  */

    if ( ! ( ReadName(Nf, Buffer, 1000, ':') ) )
    {
	Error(EOFINATT, AttName[MaxAtt], "");
    }

    MaxAttVal[MaxAtt] = 0;

    if ( Delimiter != ',' )
    {
	/*  Typed attribute  */

	if ( ! strcmp(Buffer, "continuous") )
	{
	}
	else
	if ( ! strcmp(Buffer, "timestamp") )
	{
	    SpecialStatus[MaxAtt] = TSTMPVAL;

	    /*  Set the base date if not done already  */

	    if ( ! TSBase )
	    {
		clock = time(0);
		BaseYear = gmtime(&clock)->tm_year + 1900;
		SetTSBase(BaseYear);
	    }
	}
	else
	if ( ! strcmp(Buffer, "date") )
	{
	    SpecialStatus[MaxAtt] = DATEVAL;
	}
	else
	if ( ! strcmp(Buffer, "time") )
	{
	    SpecialStatus[MaxAtt] = STIMEVAL;
	}
	else
	if ( ! memcmp(Buffer, "discrete", 8) )
	{
	    SpecialStatus[MaxAtt] = DISCRETE;

	    /*  Read max values and reserve space  */

	    v = atoi(&Buffer[8]);
	    if ( v < 2 )
	    {
		Error(BADDISCRETE, AttName[MaxAtt], "");
	    }

	    AttValName[MaxAtt] = Alloc(v+3, String);
	    AttValName[MaxAtt][0] = (char *) (long) v+1;
	    AttValName[MaxAtt][(MaxAttVal[MaxAtt]=1)] = strdup("N/A");
	}
	else
	if ( ! strcmp(Buffer, "ignore") )
	{
	    SpecialStatus[MaxAtt] = EXCLUDE;
	}
	else
	if ( ! strcmp(Buffer, "label") )
	{
	    LabelAtt = MaxAtt;
	    SpecialStatus[MaxAtt] = EXCLUDE;
	}
	else
	{
	    /*  Cannot have only one discrete value for an attribute  */

	    Error(SINGLEATTVAL, AttName[MaxAtt], Buffer);
	}
    }
    else
    {
	/*  Discrete attribute with explicit values  */

	AttValName[MaxAtt] = AllocZero(ValCeiling, String);

	/*  Add "N/A"  */

	AttValName[MaxAtt][(MaxAttVal[MaxAtt]=1)] = strdup("N/A");

	p = Buffer;

	/*  Special check for ordered attribute  */

	if ( ! memcmp(Buffer, "[ordered]", 9) )
	{
	    SpecialStatus[MaxAtt] = ORDERED;

	    for ( p = Buffer+9 ; Space(*p) ; p++ )
		;
	}

	/*  Record first real explicit value  */

	AttValName[MaxAtt][++MaxAttVal[MaxAtt]] = strdup(p);

	/*  Record remaining values  */

	do
	{
	    if ( ! ( ReadName(Nf, Buffer, 1000, ':') ) )
	    {
		Error(EOFINATT, AttName[MaxAtt], "");
	    }

	    if ( ++MaxAttVal[MaxAtt] >= ValCeiling )
	    {
		ValCeiling += 100;
		Realloc(AttValName[MaxAtt], ValCeiling, String);
	    }

	    AttValName[MaxAtt][MaxAttVal[MaxAtt]] = strdup(Buffer);
	}
	while ( Delimiter == ',' );

	/*  Cancel ordered status if <3 real values  */

	if ( Ordered(MaxAtt) && MaxAttVal[MaxAtt] <= 3 )
	{
	    SpecialStatus[MaxAtt] = 0;
	}
	if ( MaxAttVal[MaxAtt] > MaxDiscrVal ) MaxDiscrVal = MaxAttVal[MaxAtt];
    }
}
#endif



/*************************************************************************/
/*									 */
/*	Locate value Val in List[First] to List[Last]			 */
/*									 */
/*************************************************************************/


int Which(String Val, String *List, int First, int Last)
/*  -----  */
{
    int	n=First;

    while ( n <= Last && strcmp(Val, List[n]) ) n++;

    return ( n <= Last ? n : First-1 );
}



/*************************************************************************/
/*									 */
/*	Free up all space allocated by GetNames()			 */
/*									 */
/*************************************************************************/


void FreeNames()
/*   ---------  */
{
    Attribute a, t;

    ForEach(a, 1, MaxAtt)
    {
	if ( Discrete(a) )
	{
	    FreeVector((void **) AttValName[a], 1, MaxAttVal[a]);
	}
    }
    FreeUnlessNil(AttValName);				AttValName = Nil;
    FreeUnlessNil(MaxAttVal);				MaxAttVal = Nil;
    FreeVector((void **) AttName, 1, MaxAtt);		AttName = Nil;

    FreeUnlessNil(SpecialStatus);			SpecialStatus = Nil;

    /*  Definitions (if any)  */

    if ( AttDef )
    {
	ForEach(a, 1, MaxAtt)
	{
	    if ( AttDef[a] )
	    {
		for ( t = 0 ; DefOp(AttDef[a][t]) != OP_END ; t++ )
		{
		    if ( DefOp(AttDef[a][t]) == OP_STR )
		    {
			Free(DefSVal(AttDef[a][t]));
		    }
		}

		Free(AttDef[a]);
	    }
	}
	Free(AttDef);					AttDef = Nil;
    }
}



/*************************************************************************/
/*									 */
/*	Read next char keeping track of line numbers			 */
/*									 */
/*************************************************************************/


int InChar(FILE *f)
/*  ------  */
{
    if ( ! *LBp )
    {
	LBp = LineBuffer;

	if ( ! fgets(LineBuffer, MAXLINEBUFFER, f) )
	{
	    LineBuffer[0] = '\00';
	    return EOF;
	}

	LineNo++;
    }
	
    return (int) *LBp++;
}
