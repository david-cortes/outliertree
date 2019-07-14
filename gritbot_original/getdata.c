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
/*	Get case descriptions from data file				 */
/*	--------------------------------------				 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"

#define Inc 2048



/*************************************************************************/
/*									 */
/*  Read raw case descriptions from file with given extension.		 */
/*									 */
/*  On completion, cases are stored in array Case in the form		 */
/*  of descriptions (i.e. arrays of attribute values), and		 */
/*  MaxCase is set to the number of data items.				 */
/*									 */
/*************************************************************************/


void GetData(FILE *Df, Boolean Train)
/*   -------  */
{
    Description	DVec;
    CaseNo	CaseSpace;

    LineNo = 0;

    if ( Train )
    {
	MaxCase = MaxLabel = 0;
	CaseSpace = 100;
	Case = Alloc(CaseSpace+1, Description);	/* for error reporting */
    }
    else
    {
	MaxCase++;
	CaseSpace = MaxCase;
    }

    while ( (DVec = GetDescription(Df, Train)) )
    {
	/*  Make sure there is room for another item  */

	if ( MaxCase >= CaseSpace )
	{
	    CaseSpace += Inc;
	    Realloc(Case, CaseSpace+1, Description);
	}

	Case[MaxCase] = DVec;
	MaxCase++;
    }

    fclose(Df);
    MaxCase--;

}



/*************************************************************************/
/*									 */
/*  Read a raw case description from file Df.				 */
/*									 */
/*  For each attribute, read the attribute value from the file.		 */
/*  If it is a discrete valued attribute, find the associated no.	 */
/*  of this attribute value (if the value is unknown this is 0).	 */
/*									 */
/*  Returns the description of the case (i.e. a pointer to the array	 */
/*  of attribute values).						 */
/*									 */
/*************************************************************************/


Description GetDescription(FILE *Df, Boolean Train)
/*          --------------  */
{
    Attribute	Att;
    char	name[1000], *endname;
    int		Dv, Chars;
    ContValue	Cv;
    Description	DVec;
    Boolean	FirstValue=true;

    if ( ReadName(Df, name, 1000, '\00') )
    {
	Case[MaxCase] = DVec = NewCase();

	OutXVal(DVec) = 1.0;
	OutClust(DVec) = Nil;

	ForEach(Att, 1, MaxAtt)
	{
	    if ( AttDef[Att] )
	    {
		DVec[Att] = EvaluateDef(AttDef[Att], DVec);

		if ( Continuous(Att) )
		{
		    CheckValue(DVec, Att);
		}

		continue;
	    }

	    /*  Get the attribute value if don't already have it  */

	    if ( ! FirstValue && ! ReadName(Df, name, 1000, '\00') )
	    {
		Error(HITEOF, AttName[Att], "");
		FreeLastCase(DVec);
		return Nil;
	    }
	    FirstValue = false;

	    if ( Exclude(Att) )
	    {
		if ( Att == LabelAtt )
		{
		    /*  Record the value as a string  */

		    SVal(DVec,Att) = Alloc(strlen(name)+1, char);
		    strcpy(SVal(DVec,Att), name);
		}
	    }
	    else
	    if ( ! ( strcmp(name, "?") ) )
	    {
		/*  Set marker to indicate missing value  */

		DVal(DVec, Att) = UNKNOWN;
	    }
	    else
	    if ( ! strcmp(name, "N/A") )
	    {
		/*  Set marker to indicate not applicable  */

		DVal(DVec, Att) = NA;
	    }
	    else
	    if ( Discrete(Att) )
	    {
		Dv = Which(name, AttValName[Att], 1, MaxAttVal[Att]);
		if ( ! Dv )
		{
		    if ( StatBit(Att, DISCRETE) )
		    {
			if ( ! strcmp("cases", Fn + strlen(Fn) - 5) )
			{
			    /*  This is a gritcheck  */

			    Dv = UNKNOWN;
			}
			else
			{
			    /*  Add value to list  */

			    if ( MaxAttVal[Att] >= (long) AttValName[Att][0] )
			    {
				Error(TOOMANYVALS, AttName[Att],
					 (char *) AttValName[Att][0] - 1);
				Dv = MaxAttVal[Att];
			    }
			    else
			    {
				Dv = ++MaxAttVal[Att];
				AttValName[Att][Dv]   = strdup(name);
				AttValName[Att][Dv+1] = "<other>"; /* no free */
			    }

			    if ( Dv > MaxDiscrVal )
			    {
				MaxDiscrVal = Dv;
			    }
			}
		    }
		    else
		    {
			Error(BADATTVAL, AttName[Att], name);
		    }
		}
		DVal(DVec, Att) = Dv;
	    }
	    else
	    {
		/*  Continuous value  */

		if ( TStampVal(Att) )
		{
		    CVal(DVec, Att) = Cv = TStampToMins(name);
		    if ( Cv >= 1E9 )	/* long time in future */
		    {
			Error(BADTSTMP, AttName[Att], name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
		else
		if ( DateVal(Att) )
		{
		    CVal(DVec, Att) = Cv = DateToDay(name);
		    if ( Cv < 1 )
		    {
			Error(BADDATE, AttName[Att], name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
		else
		if ( TimeVal(Att) )
		{
		    CVal(DVec, Att) = Cv = TimeToSecs(name);
		    if ( Cv < 0 )
		    {
			Error(BADTIME, AttName[Att], name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}
		else
		{
		    CVal(DVec, Att) = strtod(name, &endname);
		    if ( endname == name || *endname != '\0' )
		    {
			Error(BADATTVAL, AttName[Att], name);
			DVal(DVec, Att) = UNKNOWN;
		    }
		}

		CheckValue(DVec, Att);
	    }
	}

	if ( LabelAtt && (Chars = strlen(SVal(DVec, LabelAtt))) > MaxLabel )
	{
	    MaxLabel = Chars;
	}

	return DVec;
    }
    else
    {
	return Nil;
    }
}



/*************************************************************************/
/*									 */
/*	Free case description space					 */
/*									 */
/*************************************************************************/


void FreeData()
/*   --------  */
{
    CaseNo	i;

    /*  Release any strings holding case labels  */

    if ( LabelAtt )
    {
	ForEach(i, 0, MaxCase)
	{
	    FreeUnlessNil(SVal(Case[i],LabelAtt));
	}
    }

    FreeCases();

    Free(Case);						Case = Nil;
    MaxCase = -1;
}



/*************************************************************************/
/*									 */
/*	Check for bad continuous value					 */
/*									 */
/*************************************************************************/


void CheckValue(Description DVec, Attribute Att)
/*   ----------  */
{
    ContValue	Cv;

    Cv = CVal(DVec, Att);
    if ( ! finite(Cv) )
    {
	Error(BADNUMBER, AttName[Att], "");

	CVal(DVec, Att) = UNKNOWN;
	DVal(DVec, Att) = 0;
    }
}
