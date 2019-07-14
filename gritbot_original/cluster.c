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
/*	Routines to manage clusters					 */
/*	---------------------------					 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"


/*************************************************************************/
/*									 */
/*	Register a new cluster						 */
/*									 */
/*************************************************************************/


Clust NewClust(ContValue Expect, ContValue SD, ContValue Limit,
	       CaseCount Anoms, CaseCount GpSize)
/*    --------  */
{
    Clust	C;

    /*  Make sure we have room for another  */

    if ( NClust >= ClustSpace )
    {
	Realloc(Cluster, (ClustSpace += 1000), Clust);
    }

    C = Cluster[NClust++] = Alloc(1, ClustRec);

    /*  Save cluster information  */

    C->Att    = ClassAtt;
    C->Expect = Expect;
    C->SD     = SD;
    C->Limit  = Limit;
    C->GpSize = GpSize;
    C->Frac   = 1 - Anoms / (float) GpSize;

    SaveClustConds(C);

    return C;
}



/*************************************************************************/
/*									 */
/*	Process current tests to determine cluster conditions.		 */
/*	Different functions are called depending on the condition	 */
/*	type.								 */
/*									 */
/*************************************************************************/


void SaveClustConds(Clust C)
/*   --------------  */
{
    Attribute	Att, NAtts=0;
    int		NC=0;

    /*  Count attributes tested  */

    ForEach(Att, 1, MaxAtt)
    {
	if ( GEnv.Tested[Att] ) NAtts++;
    }

    C->NCond = NAtts;
    C->Cond  = Alloc(C->NCond, ClustCond);

    /*  Format tests on each attribute  */

    ForEach(Att, 1, MaxAtt)
    {
	if ( GEnv.Tested[Att] )
	{
	    if ( Continuous(Att) )
	    {
		FormatContinCond(Att, &C->Cond[NC]);
	    }
	    else
	    if (  Ordered(Att) )
	    {
		FormatOrderedCond(Att, &C->Cond[NC]);
	    }
	    else
	    if ( Continuous(ClassAtt) && MaxAttVal[Att] > 3 )
	    {
		FormatSubsetCond(Att, &C->Cond[NC]);
	    }
	    else
	    {
		FormatValCond(Att, &C->Cond[NC]);
	    }

	    NC++;
	}
    }
}



/*************************************************************************/
/*									 */
/*	Test on a continuous att.  Check all threshold tests		 */
/*	and assemble lowest and highest possible value			 */
/*									 */
/*************************************************************************/


void FormatContinCond(Attribute Att, ClustCond *CC)
/*   ----------------  */
{
    ContValue	Lo=-MARKER, Hi=MARKER;
    int		i, Type=0;

    ForEach(i, 0, GEnv.Level)
    {
	if ( GEnv.Test[i].Att == Att )
	{
	    if ( GEnv.Test[i].Br == 1 )
	    {
		Type = CONT_NA;
		Lo = 1;
		Hi = 0;
		break;
	    }
	    else
	    if ( GEnv.Test[i].Br == 2 )
	    {
		Type |= CONT_LT;
		Hi = GEnv.Test[i].Cut;
	    }
	    else
	    {
		Type |= CONT_GT;
		Lo = GEnv.Test[i].Cut;
	    }
	}
    }

    CC->Type   = Type;
    CC->Att    = Att;
    CC->Low    = Lo;
    CC->High   = Hi;
    CC->Values = Nil;
}



/*************************************************************************/
/*									 */
/*	Test on an ordered discrete attribute (similar to above)	 */
/*									 */
/*************************************************************************/


void FormatOrderedCond(Attribute Att, ClustCond *CC)
/*   -----------------  */
{
    DiscrValue	Lo, Hi;
    int		i, Type=0;

    Lo = 2;
    Hi = MaxAttVal[Att];

    ForEach(i, 0, GEnv.Level)
    {
	if ( GEnv.Test[i].Att == Att )
	{
	    if ( GEnv.Test[i].Br == 1 )
	    {
		Type = DISCR_VAL;
		Lo = Hi = 1;
		break;
	    }
	    else
	    if ( GEnv.Test[i].Br == 2 )
	    {
		Type |= DISCR_LT;
		Hi = GEnv.Test[i].Cut;
	    }
	    else
	    {
		Type |= DISCR_GT;
		Lo = GEnv.Test[i].Cut + 1;
	    }
	}
    }

    CC->Type   = Type;
    CC->Att    = Att;
    CC->Low    = Lo;
    CC->High   = Hi;
    CC->Values = Nil;
}



/*************************************************************************/
/*									 */
/*	Subset test for a discrete attribute.  All tests must be	 */
/*	checked to determine the final subset values			 */
/*									 */
/*************************************************************************/


void FormatSubsetCond(Attribute Att, ClustCond *CC)
/*   ----------------  */
{
    DiscrValue	v;
    int		i;

    CC->Att = Att;

    GEnv.Possible[1] = false;
    ForEach(v, 2, MaxAttVal[Att])
    {
	GEnv.Possible[v] = true;
    }

    ForEach(i, 0, GEnv.Level)
    {
	if ( GEnv.Test[i].Att == Att )
	{
	    if ( GEnv.Test[i].Br == 1 )
	    {
		GEnv.Possible[1] = true;
		ForEach(v, 2, MaxAttVal[Att])
		{
		    GEnv.Possible[v] = false;
		}
		break;
	    }
	    else
	    ForEach(v, 2, MaxAttVal[Att])
	    {
		if ( In(v, GEnv.Test[i].Left) )
		{
		    GEnv.Possible[v] = GEnv.Possible[v] && ( GEnv.Test[i].Br == 2 );
		}
		else
		{
		    GEnv.Possible[v] = GEnv.Possible[v] && ( GEnv.Test[i].Br == 3 );
		}
	    }
	}
    }

    CC->Type   = DISCR_SET;
    CC->Low    = CC->High = 0;
    CC->Values = AllocZero((MaxAttVal[Att]>>3)+1, unsigned char);

    ForEach(v, 1, MaxAttVal[Att])
    {
	if ( GEnv.Possible[v] ) SetBit(v, CC->Values);
    }
}



/*************************************************************************/
/*									 */
/*	Simple test on attribute value.  There is no need to check	 */
/*	more than one test since the first determines the tested value	 */
/*									 */
/*************************************************************************/


void FormatValCond(Attribute Att, ClustCond *CC)
/*   -------------  */
{
    int		i;

    ForEach(i, 0, GEnv.Level)
    {
	if ( GEnv.Test[i].Att == Att )
	{
	    CC->Type   = DISCR_VAL;
	    CC->Att    = Att;
	    CC->Low    = CC->High = GEnv.Test[i].Br;
	    CC->Values = Nil;
	    return;
	}
    }
}



/*************************************************************************/
/*									 */
/*	Free conditions stored in a cluster				 */
/*									 */
/*************************************************************************/


void FreeClust(Clust C)
/*   ---------  */
{
    int		d;

    if ( C )
    {
	ForEach(d, 0, C->NCond-1)
	{
	    FreeUnlessNil(C->Cond[d].Values);
	}
	FreeUnlessNil(C->Cond);
	Free(C);
    }
}
