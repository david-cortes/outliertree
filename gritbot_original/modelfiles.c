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
/*	Routines for saving and reading sift files			 */
/*	------------------------------------------			 */
/*									 */
/*************************************************************************/

/*  Formats:

    1 <att>						target att
    2 <att>						use logs with att
    3 <att> <ltail> <htail>				set low/high tail

    11 <lev> <att> <br>					split on attribute value
    12 <lev> <att> <br> <cut>				threshold test
    13 <lev> <att> <br> <subset>			subset test

    21 <cov> <frac> <mode> [<val> <cf>]* 0		discrete cluster
    22 <cov> <mean> <sd> <lfrac> <llim> <hfrac> <hlim>	low/high contin cluster

	clusters may be followed by one or more caveats of the form
	    <att> <low> <high> or
	    <att> <subset>
*/

#include "defns.i"
#include "extern.i"

int	Entry;

char*	Prop[]={"null",
		"id",
		"atts",
		"att",
		"sstat",
		"elts",
		"prec",
		"label",
		"def",
		"minabnorm"
	       };

char	PropName[20],
	*PropVal=Nil,
	*Unquoted;
int	PropValSize=0;

#define	PROPS 9

#define ERRORP		0
#define	IDP		1
#define	ATTSP		2
#define	ATTP		3
#define	SSTATP		4
#define	ELTSP		5
#define	PRECP		6
#define	LABELP		7
#define	DEFP		8
#define	MINABNORMP	9


/*************************************************************************/
/*									 */
/*	Check whether file is open.  If it is not, open it and		 */
/*	read/write header information and names				 */
/*									 */
/*************************************************************************/


void CheckFile(String Extension, Boolean Write)
/*   ---------  */
{
    static char	*LastExt="";

    if ( ! Sf || strcmp(LastExt, Extension) )
    {
	LastExt = Extension;

	if ( Sf )
	{
	    fprintf(Sf, "\n");
	    fclose(Sf);
	}

#ifdef INSPECT
	ReadFilePrefix(Extension);
#else
	if ( Write )
	{
	    WriteFilePrefix(Extension);
	}
	else
	{
	    ReadFilePrefix(Extension);
	}
#endif
    }
}
#ifndef INSPECT



/*************************************************************************/
/*									 */
/*	Write information on system, parameters etc			 */
/*									 */
/*************************************************************************/


void WriteFilePrefix(String Extension)
/*   ---------------  */
{
    time_t	clock;
    struct tm	*now;

    if ( ! (Sf = GetFile(Extension, "w")) )
    {
	Error(NOFILE, Fn, E_ForWrite);
    }

    clock = time(0);
    now = localtime(&clock);
    now->tm_mon++;
    fprintf(Sf, "id=\"GritBot %s %d-%d%d-%d%d\"\n",
	    RELEASE,
	    now->tm_year + 1900,
	    now->tm_mon / 10, now->tm_mon % 10,
	    now->tm_mday / 10, now->tm_mday % 10);

    SaveNames();

    fprintf(Sf, "minabnorm=\"%g\"\n", MINABNORM);
}



void SaveCondition()
/*   --------------  */
{
    Attribute	Att;
    int		CType, b, Bytes;
    DiscrValue	Br;
    char	SE[100];

    if ( GEnv.Level < 0 ) return;

    /*  Save top condition */

    Att = GEnv.Test[GEnv.Level].Att;
    Br  = GEnv.Test[GEnv.Level].Br;

    /*  Determine condition type  */

    CType = ( Br == 1 ? 11 :
	      Continuous(Att) || Ordered(Att) ? 12 :
	      Continuous(ClassAtt) && MaxAttVal[Att] > 3 ? 13 : 11 );

    sprintf(SE, "%d %d %d %d", CType, GEnv.Level, Att, Br);
    ExtendSiftEntry(SE);

    /*  Don't need to save anything else if this branch is 1 (N/A)
	or if test is on two-valued discrete att  */

    if ( Br != 1 )
    {
	if ( Continuous(Att) || Ordered(Att) )
	{
	    sprintf(SE, " %.8g", GEnv.Test[GEnv.Level].Cut);
	    ExtendSiftEntry(SE);
	}
	else
	if ( Continuous(ClassAtt) && MaxAttVal[Att] > 3 )
	{
	    /*  Print subset of values  */

	    Bytes = (MaxAttVal[Att]>>3) + 1;

	    ForEach(b, 0, Bytes-1)
	    {
		sprintf(SE, " %x", GEnv.Test[GEnv.Level].Left[b]);
		ExtendSiftEntry(SE);
	    }
	}
    }

    ExtendSiftEntry("\n");
}



void SaveDiscrCluster(DiscrValue Expect, CaseCount Anoms, CaseCount Cases,
		      CaseCount *Freq)
/*   ----------------  */
{
    DiscrValue	v;
    float	Xv;
    char	SE[100];

    SaveCondition();

    sprintf(SE, "21 %d %g %d",
		Cases, (Cases - Anoms) / (Cases + 1E-3), Expect);
    ExtendSiftEntry(SE);

    ForEach(v, 1, MaxAttVal[ClassAtt])
    {
	if ( v == Expect ) continue;

	Xv = XDScore(Freq[v], Cases, Anoms, Prior[ClassAtt][v]);

	if ( Xv <= 1.0 / (MINABNORM * MINABNORM) )
	{
	    sprintf(SE, " %d %.3f", v, Xv);
	    ExtendSiftEntry(SE);
	}
    }
    ExtendSiftEntry(" 0");
}



void SaveContinCluster(float Mean, float SD, CaseCount Cases,
		       float LFrac, float LLim, float HFrac, float HLim)
/*   -----------------  */
{
    char	SE[100];

    SaveCondition();

    sprintf(SE, "22 %d %g %g %g %.8g %g %.8g",
		Cases, Mean, SD, LFrac, LLim, HFrac, HLim);
    ExtendSiftEntry(SE);
}



/*************************************************************************/
/*									 */
/*	Save attribute information					 */
/*									 */
/*************************************************************************/


void SaveNames()
/*   ---------  */
{
    Attribute	Att;
    DiscrValue	v;
    int		DN, Op;

    fprintf(Sf, "atts=\"%d\"\n", MaxAtt);

    ForEach(Att, 1, MaxAtt)
    {
	AsciiOut("att=", AttName[Att]);
	fprintf(Sf, " sstat=\"%d\"", SpecialStatus[Att]);

	if ( AttDef[Att] )
	{
	    /*  Dump definition  */

	    fprintf(Sf, " def");
	    for ( DN = 0 ; ; DN++ )
	    {
		Op = DefOp(AttDef[Att][DN]);

		fprintf(Sf, "%c\"%d\"", ( DN ? ',' : '=' ), Op);
		if ( Op == OP_ATT )
		{
		    fprintf(Sf, ":\"%d\"",
				(int) (long) DefSVal(AttDef[Att][DN]));
		}
		else
		if ( Op == OP_STR )
		{
		    AsciiOut(":", DefSVal(AttDef[Att][DN]));
		}
		else
		{
		    fprintf(Sf, ":\"%g\"", DefNVal(AttDef[Att][DN]));
		}

		if ( Op == OP_END ) break;
	    }
	}

	if ( MaxAttVal[Att] > 0 )
	{
	    AsciiOut(" elts=", AttValName[Att][2]); 	/* skip N/A */

	    ForEach(v, 3, MaxAttVal[Att])
	    {
		AsciiOut(",", AttValName[Att][v]);
	    }
	}
	
	if ( Prec[Att] > 0 )
	{
	    fprintf(Sf, " prec=\"%d\"", Prec[Att]);
	}

	fprintf(Sf, "\n");
    }

    if ( LabelAtt )
    {
	AsciiOut("label=", AttName[LabelAtt]);
	fprintf(Sf, "\n");
    }
}



/*************************************************************************/
/*									 */
/*	Write ASCII string with prefix, escaping any quotes		 */
/*									 */
/*************************************************************************/


void AsciiOut(String Pre, String S)
/*   --------  */
{
    fprintf(Sf, "%s\"", Pre);
    while ( *S )
    {
	if ( *S == '"' || *S == '\\' ) fputc('\\', Sf);
	fputc(*S++, Sf);
    }
    fputc('"', Sf);
}
#endif



/*************************************************************************/
/*									 */
/*	Read header information						 */
/*									 */
/*************************************************************************/


void ReadFilePrefix(String Extension)
/*   --------------  */
{
    Attribute	Att=0;
    DiscrValue	v;
    char	*p, Dummy;
    int		Year, Month, Day, X, Elts, Op, DN, DefSize;
    float	A;
    extern AttValue	_UNK, _NA;

    if ( ! (Sf = GetFile(Extension, "r")) ) Error(NOFILE, Fn, "");

    while ( true )
    {
	switch ( ReadProp(&Dummy) )
	{
	    case ERRORP:
		Error(BADSIFT, PropName, "");
		return;

	    case IDP:
		/*  Recover year run and set base date for timestamps  */

		if ( sscanf(PropVal + strlen(PropVal) - 11,
			    "%d-%d-%d\"", &Year, &Month, &Day) == 3 )
		{
		    SetTSBase(Year);
		}
		break;

	    case ATTSP:
		Unquoted = RemoveQuotes(PropVal);
		if ( sscanf(Unquoted, "%d", &MaxAtt) != 1 )
		{
		    Error(BADSIFT, "atts", "");
		}

		AttName	      = Alloc(MaxAtt+1, String);
		SpecialStatus = AllocZero(MaxAtt+1, char);
		AttDef        = AllocZero(MaxAtt+1, Definition);
		AttValName    = Alloc(MaxAtt+1, String *);
		MaxAttVal     = AllocZero(MaxAtt+1, int);
		Prec          = Alloc(MaxAtt+1, unsigned char);
		UseLogs       = AllocZero(MaxAtt+1, Boolean);
		LowTail	      = Alloc(MaxAtt+1, float);
		HighTail      = Alloc(MaxAtt+1, float);

		Att = 0;
		break;

	    case ATTP:
		Att++;
		if ( Att > MaxAtt )
		{
		    Error(BADSIFT, "att", "");
		}

		Unquoted = RemoveQuotes(PropVal);
		AttName[Att] = strdup(Unquoted);

		LowTail[Att]  = -1E38;
		HighTail[Att] = 1E38;
		break;

	    case SSTATP:
		Unquoted = RemoveQuotes(PropVal);
		if ( sscanf(Unquoted, "%d", &X) != 1 )
		{
		    Error(BADSIFT, "sstat", "");
		}
		SpecialStatus[Att] = X;
		break;

	    case ELTSP:
		Elts = 100;
		AttValName[Att] = Alloc(Elts, String);

		MaxAttVal[Att] = 1;
		AttValName[Att][1] = strdup("N/A");

		for ( p = PropVal ; *p ; )
		{
		    p = RemoveQuotes(p);
		    v = ++MaxAttVal[Att];

		    if ( v+2 >= Elts )
		    {
			Elts += 100;
			Realloc(AttValName[Att], Elts, String);
		    }

		    AttValName[Att][v] = strdup(p);

		    for ( p += strlen(p) ; *p != '"' ; p++ )
			;
		    p++;
		    if ( *p == ',' ) p++;
		}
		break;

	    case PRECP:
		Unquoted = RemoveQuotes(PropVal);
		if ( sscanf(Unquoted, "%d", &X) != 1 )
		{
		    Error(BADSIFT, "prec", "");
		}
		Prec[Att] = X;
		break;

	    case LABELP:
		Unquoted = RemoveQuotes(PropVal);
		LabelAtt = Which(Unquoted, AttName, 1, MaxAtt);
		break;

	    case DEFP:
		/*  Make sure _UNK and _NA are set  */

		_UNK._discr_val = UNKNOWN;
		_NA._discr_val  = NA;

		/*  Allocate initial space for definition  */

		AttDef[Att] = AllocZero(DefSize = 100, DefElt);
		DN = 0;

		/*  Read definition operators  */

		for ( p = PropVal ; *p ; )
		{
		    /*  Check that space is available  */

		    if ( DN >= DefSize )
		    {
			DefSize += 100;
			Realloc(AttDef[Att], DefSize, DefElt);
		    }

		    /*  Read opcode  */

		    p = RemoveQuotes(p);
		    if ( sscanf(p, "%d", &Op) != 1 )
		    {
			Error(BADSIFT, "def", "");
		    }
		    DefOp(AttDef[Att][DN]) = Op;

		    /*  Move to start of operand  */

		    for ( p += strlen(p) ; *p != '"' ; p++ )
			;
		    p++;
		    if ( *p != ':' )
		    {
			Error(BADSIFT, "def", "");
		    }
		    p++;

		    /*  Read operand -- depends on opcode  */

		    p = RemoveQuotes(p);

		    if ( Op == OP_ATT )
		    {
			if ( sscanf(p, "%d", &X) != 1 )
			{
			    Error(BADSIFT, "def", "");
			}
			DefSVal(AttDef[Att][DN]) = (void *) X;
		    }
		    else
		    if ( Op == OP_STR )
		    {
			DefSVal(AttDef[Att][DN]) = strdup(p);
		    }
		    else
		    {
			if ( sscanf(p, "%g", &A) != 1 )
			{
			    Error(BADSIFT, "def", "");
			}
			DefNVal(AttDef[Att][DN]) = A;
		    }

		    /*  Get ready for next entry  */

		    DN++;

		    for ( p += strlen(p) ; *p != '"' ; p++ )
			;
		    p++;
		    if ( *p == ',' ) p++;
		}
		break;

	    case MINABNORMP:
		Unquoted = RemoveQuotes(PropVal);
		if ( sscanf(Unquoted, "%g", &MINABNORM) != 1 )
		{
		    Error(BADSIFT, "minabnorm", "");
		}
		return;
	}
    }
}



/*************************************************************************/
/*									 */
/*	ASCII reading utilities						 */
/*									 */
/*************************************************************************/


int ReadProp(char *Delim)
/*  --------  */
{
    int		c, i;
    char	*p;
    Boolean	Quote=false;

    for ( p = PropName ; (c = fgetc(Sf)) != '=' ;  )
    {
	if ( p - PropName >= 19 || c == EOF )
	{
	    Error(BADSIFT, "EOF", "");
	    PropName[0] = PropVal[0] = *Delim = '\00';
	    return 0;
	}
	*p++ = c;
    }
    *p = '\00';

    for ( p = PropVal ; ((c = fgetc(Sf)) != ' ' && c != '\n') || Quote ; )
    {
	if ( c == EOF )
	{
	    Error(BADSIFT, "EOF", "");
	    PropName[0] = PropVal[0] = '\00';
	    return 0;
	}

	if ( (i = p - PropVal) >= PropValSize )
	{
	    PropValSize += 10000;
	    Realloc(PropVal, PropValSize + 3, char);
	    p = PropVal + i;
	}

	*p++ = c;
	if ( c == '\\' )
	{
	    *p++ = fgetc(Sf);
	}
	else
	if ( c == '"' )
	{
	    Quote = ! Quote;
	}
    }
    *p = '\00';
    *Delim = c;

    return Which(PropName, Prop, 1, PROPS);
}



String RemoveQuotes(String S)
/*     ------------  */
{
    char	*p, *Start;

    p = Start = S;
    
    for ( S++ ; *S != '"' ; S++ )
    {
	if ( *S == '\\' ) S++;
	*p++ = *S;
	*S = '-';
    }
    *p = '\00';

    return Start;
}



/*************************************************************************/
/*									 */
/*	Add more text to a sift entry					 */
/*									 */
/*************************************************************************/


void ExtendSiftEntry(String S)
/*   ---------------  */
{
    int Len;

    /*  Make sure there is enough room  */

    if ( (Len = strlen(S)) >= GEnv.SiftSpace - GEnv.SiftSize )
    {
	GEnv.SiftSpace += 1000;

	if ( GEnv.SiftEntry )
	{
	     Realloc(GEnv.SiftEntry, GEnv.SiftSpace, char);
	}
	else
	{
	    GEnv.SiftEntry = Alloc(GEnv.SiftSpace, char);
	}
    }

    strcpy(GEnv.SiftEntry + GEnv.SiftSize, S);
    GEnv.SiftSize += Len;
}
