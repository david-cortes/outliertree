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
/*								 	 */
/*    Divide-and-Conquer for discrete attributes			 */
/*    ------------------------------------------			 */
/*								 	 */
/*************************************************************************/

#include "defns.i"
#include "extern.i"


/*************************************************************************/
/*								  	 */
/*	Known values of continuous attributes are divided into		 */
/*	three groups:							 */
/*	  (1)	N/A							 */
/*	  (2)   values less than a threshold				 */
/*	  (3)	values greater than a threshold				 */
/*	This routine finds the best threshold for items Fp through Lp	 */
/*	and sets Gain[], Info[] and Bar[]				 */
/*								  	 */
/*************************************************************************/


void DEvalContinAtt(Attribute Att, CaseNo Fp, CaseNo Lp)
/*   --------------  */
{
    CaseNo	i, BestI, Xp;
    CaseCount	Cases;
    DiscrValue	v, c;
    double	BestGain=-1E-6, BestInfo, ThisGain;

    /*  Reset frequencies  */

    ForEach(v, 0, 3)
    {
	ForEach(c, 1, MaxAttVal[ClassAtt])
	{
	    GEnv.Freq[v][c] = 0;
	}
	GEnv.ValFreq[v] = 0;
    }

    /*  Omit and count N/A values */

    Xp = Fp;
    ForEach(i, Fp, Lp)
    {
	if ( NotApplic(Case[i],Att) )
	{
	    GEnv.Freq[1][DClass(Case[i])]++;
	    Swap(Xp, i);
	    Xp++;
	}
	else
	{
	    GEnv.Freq[3][DClass(Case[i])]++;
	}
    }

    /*  Special case when very few known values  */

    if ( No(Xp, Lp) < 2 * (DMINITEMS*GEnv.FRAC) )
    {
	Verbosity(2,
	    fprintf(Of, "\tAtt %s: insufficient cases with known values\n",
			AttName[Att]))
	return;
    }

    Cases = No(Fp, Lp);

    GEnv.ValFreq[0] = GEnv.ValFreq[1] = 0;
    ForEach(c, 1, MaxAttVal[ClassAtt])
    {
	GEnv.ValFreq[1] += GEnv.Freq[1][c];
    }

    /*  Sort all applicable values  */

    Quicksort(Xp, Lp, Att);

    /*  Try possible cuts between items i and i+1, and determine the
	information and gain of the split in each case  */

    if ( Xp + (DMINITEMS*GEnv.FRAC) - 2 >= Lp ) return;
    ForEach(i, Xp, Xp + (DMINITEMS*GEnv.FRAC) - 2)
    {
	c = DClass(Case[i]);

	GEnv.Freq[2][c]++;
	GEnv.Freq[3][c]--;
    }

    ForEach(i, Xp + (DMINITEMS*GEnv.FRAC) - 1, Lp - (DMINITEMS*GEnv.FRAC))
    {
	c = DClass(Case[i]);

	GEnv.Freq[2][c]++;
	GEnv.Freq[3][c]--;

	if ( CVal(Case[i+1], Att) > CVal(Case[i], Att) )
	{
	    GEnv.ValFreq[2] = i - Xp + 1;
	    GEnv.ValFreq[3] = Lp - i;

	    ThisGain = DiscrGain(3, Cases);

	    if ( ThisGain > BestGain + Epsilon )
	    {
		BestGain = ThisGain;
		BestInfo = TotalInfo(GEnv.ValFreq, 1, 3) / Cases;
		BestI    = i;

		ForEach(v, 1, 3)
		{
		    ForEach(c, 1, MaxAttVal[ClassAtt])
		    {
			GEnv.BestFreq[v][c] = GEnv.Freq[v][c];
		    }
		}
	    }
	}
    }

    /*  If a test on the attribute is able to make a gain,
	set the best break point, gain and information  */

    if ( BestGain > Epsilon )
    {
	GEnv.Gain[Att] = BestGain;
	GEnv.Info[Att] = BestInfo;
	GEnv.Bar[Att]  = Between(CVal(Case[BestI],Att), CVal(Case[BestI+1],Att));

	Verbosity(2,
	    fprintf(Of, "\tAtt %s: cut=%.3f, inf %.3f, gain %.3f\n",
		   AttName[Att], GEnv.Bar[Att], GEnv.Info[Att], GEnv.Gain[Att]))

	/*  If not sampling, check subsets  */

	if ( GEnv.FRAC >= 1 )
	{
	    if ( Xp > Fp )
	    {
		NoteTest(Att, 1, 0.0, Nil);
		FindDiscrOutliers(Fp, Xp-1, GEnv.BestFreq[1]);
	    }

	    NoteTest(Att, 2, GEnv.Bar[Att], Nil);
	    FindDiscrOutliers(Xp, BestI, GEnv.BestFreq[2]);

	    NoteTest(Att, 3, GEnv.Bar[Att], Nil);
	    FindDiscrOutliers(BestI+1, Lp, GEnv.BestFreq[3]);
	}
    }
    else
    {
	Verbosity(2, fprintf(Of, "\tAtt %s: no gain\n", AttName[Att]))
    }
}



/*************************************************************************/
/*									 */
/*	Set Info[] and Gain[] for discrete partition of items Fp to Lp	 */
/*									 */
/*************************************************************************/


void DEvalDiscrAtt(Attribute Att, CaseNo Fp, CaseNo Lp)
/*   -------------  */
{
    CaseCount	KnownCases;
    int		ReasonableSubsets=0;
    DiscrValue	v;

    ComputeFrequencies(Att, Fp, Lp);
    KnownCases = No(Fp, Lp);

    /*  Check reasonable subsets  */

    ForEach(v, 1, MaxAttVal[Att])
    {
	if ( GEnv.ValFreq[v] >= (DMINITEMS*GEnv.FRAC) ) ReasonableSubsets++;
    }

    if ( ReasonableSubsets < 2 )
    {
	Verbosity(2, fprintf(Of, "\tAtt %s: poor split\n", AttName[Att]))
	return;
    }

    GEnv.Gain[Att] = DiscrGain(MaxAttVal[Att], KnownCases);
    GEnv.Info[Att] = TotalInfo(GEnv.ValFreq, 1, MaxAttVal[Att]) / KnownCases;

    if ( GEnv.Gain[Att] > Epsilon )
    {
	Verbosity(2,
	    fprintf(Of, "\tAtt %s: inf %.3f, gain %.3f\n",
			AttName[Att], GEnv.Info[Att], GEnv.Gain[Att]))
    }
    else
    {
	GEnv.Gain[Att] = None;
	Verbosity(2,
	    fprintf(Of, "\tAtt %s: no gain\n", AttName[Att]))
    }

    if ( GEnv.FRAC >= 1 && GEnv.Gain[Att] > Epsilon )
    {
	CheckPotentialClusters(Att, MaxAttVal[Att], Fp, Lp, 0.0, Nil, GEnv.Freq);
    }
}



/*************************************************************************/
/*									 */
/*	Set Info[] and Gain[] for ordered split on items Fp to Lp	 */
/*									 */
/*************************************************************************/


void DEvalOrderedAtt(Attribute Att, CaseNo Fp, CaseNo Lp)
/*   ---------------  */
{
    CaseCount	KnownCases, *HoldFreqRow, SplitFreq[4];
    DiscrValue	v, BestV, vv, c;
    double	ThisGain, BestInfo, BestGain=-1E-6;

    ComputeFrequencies(Att, Fp, Lp);

    KnownCases = No(Fp, Lp);

    /*  Move elts of Freq[] starting with the third up one place
	and aggregate class frequencies  */

    HoldFreqRow = GEnv.Freq[MaxAttVal[Att]+1];
    ForEach(c, 1, MaxAttVal[ClassAtt])
    {
	HoldFreqRow[c] = 0;
    }
    SplitFreq[0] = GEnv.ValFreq[0];
    SplitFreq[1] = GEnv.ValFreq[1];
    SplitFreq[2] = GEnv.ValFreq[2];
    SplitFreq[3] = 0;

    for ( v = MaxAttVal[Att] ; v > 2 ; v-- )
    {
	GEnv.Freq[v+1] = GEnv.Freq[v];
	ForEach(c, 1, MaxAttVal[ClassAtt])
	{
	    HoldFreqRow[c] += GEnv.Freq[v][c];
	}
	SplitFreq[3] += GEnv.ValFreq[v];
    }

    GEnv.Freq[3] = HoldFreqRow;

    /*  Try various cuts, saving the one with maximum gain  */

    ForEach(v, 3, MaxAttVal[Att])
    {
	if ( SplitFreq[2] >= (DMINITEMS*GEnv.FRAC) &&
	     SplitFreq[3] >= (DMINITEMS*GEnv.FRAC) )
	{
	    ThisGain = DiscrGain(3, KnownCases);

	    if ( ThisGain > BestGain + Epsilon )
	    {
		BestGain = ThisGain;
		BestInfo = TotalInfo(SplitFreq, 0, 3) / KnownCases;
		BestV    = v-1;

		ForEach(vv, 1, 3)
		{
		    ForEach(c, 1, MaxAttVal[ClassAtt])
		    {
			GEnv.BestFreq[vv][c] = GEnv.Freq[vv][c];
		    }
		}
	    }
	}

	/*  Move val v from right branch to left branch  */

	ForEach(c, 1, MaxAttVal[ClassAtt])
	{
	    GEnv.Freq[2][c] += GEnv.Freq[v+1][c];
	    GEnv.Freq[3][c] -= GEnv.Freq[v+1][c];
	}
	SplitFreq[2] += GEnv.ValFreq[v];
	SplitFreq[3] -= GEnv.ValFreq[v];
    }

    /*  If a test on the attribute is able to make a gain,
	set the best break point, gain and information  */

    if ( BestGain > Epsilon )
    {
	GEnv.Gain[Att] = BestGain;
	GEnv.Info[Att] = BestInfo;
	GEnv.Bar[Att]  = BestV + 0.1;

	ClearBits((MaxAttVal[Att]>>3)+1, GEnv.Subset[Att]);
	ForEach(v, 2, BestV)
	{
	    SetBit(v, GEnv.Subset[Att]);
	}

	Verbosity(2,
	    fprintf(Of, "\tAtt %s: cut after %s, inf %.3f, gain %.3f\n",
			AttName[Att], AttValName[Att][(int) GEnv.Bar[Att]],
			GEnv.Info[Att], GEnv.Gain[Att]))

	if ( GEnv.FRAC >= 1 )
	{
	    CheckPotentialClusters(Att, 3, Fp, Lp, GEnv.Bar[Att], GEnv.Subset[Att],
				   GEnv.BestFreq);
	}
    }
    else
    {
	Verbosity(2, fprintf(Of, "\tAtt %s: no gain\n", AttName[Att]))
    }
}



/*************************************************************************/
/*									 */
/*	Compute frequency tables Freq[][] and ValFreq[] for attribute	 */
/*	Att from items Fp to Lp						 */
/*									 */
/*************************************************************************/


void ComputeFrequencies(Attribute Att, CaseNo Fp, CaseNo Lp)
/*   ------------------  */
{
    DiscrValue	v, c;
    CaseNo	Sum;

    ForEach(v, 0, MaxAttVal[Att])
    {
	Sum = 0;
	ForEach(c, 1, MaxAttVal[ClassAtt])
	{
	    Sum += (GEnv.Freq[v][c] = GEnv.DFreq[Att][v][c]);
	}
	GEnv.ValFreq[v] = Sum;
    }
}



void FindClassFrequencies(CaseNo Fp, CaseNo Lp)
/*   --------------------  */
{
    DiscrValue	v;
    CaseNo	i;

    ForEach(v, 1, MaxAttVal[ClassAtt])
    {
	GEnv.ClassFreq[v] = 0;
    }

    ForEach(i, Fp, Lp)
    {
	GEnv.ClassFreq[DClass(Case[i])]++;
    }
}



/*************************************************************************/
/*									 */
/*	Given Freq[][] and ValFreq[], compute the information gain	 */
/*									 */
/*************************************************************************/


double DiscrGain(DiscrValue MaxVal, CaseCount KnownCases)
/*     ---------  */
{
    DiscrValue	v;
    double	ThisInfo=0.0;

    /*  Check whether all values are unknown or the same  */

    if ( ! KnownCases ) return None;

    /*  Compute total info after split, by summing the
	info of each of the subsets formed by the test  */

    ForEach(v, 1, MaxVal)
    {
	ThisInfo += TotalInfo(GEnv.Freq[v], 1, MaxAttVal[ClassAtt]);
    }

    /*  Set the gain in information for all items */

    return GEnv.BaseInfo - ThisInfo / KnownCases;
}



/*************************************************************************/
/*									 */
/*	Compute the total information in V[ MinVal..MaxVal ].		 */
/*	Use tabulate logs of numbers of cases				 */
/*									 */
/*************************************************************************/


double TotalInfo(CaseCount V[], DiscrValue MinVal, DiscrValue MaxVal)
/*     ---------  */
{
    DiscrValue	v;
    double	Sum=0.0;
    CaseCount	N, TotalCases=0;

    ForEach(v, MinVal, MaxVal)
    {
	N = V[v];

	Sum += N * LogCaseNo[N];
	TotalCases += N;
    }

    return TotalCases * LogCaseNo[TotalCases] - Sum;
}
