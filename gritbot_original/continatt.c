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
/*    Divide-and-Conquer for continuous attributes			 */
/*    --------------------------------------------			 */
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
/*	and sets Gain[] and Bar[]					 */
/*								  	 */
/*************************************************************************/


void CEvalContinAtt(Attribute Att, CaseNo Fp, CaseNo Lp)
/*   --------------  */ 
{ 
    CaseNo	i, BestI, Xp;
    double	Val, ThisGain, BestGain=-1E-6;

    /*  Special case when very few values  */

    if ( No(Fp, Lp) < 2 * (CMINITEMS*GEnv.FRAC) )
    {
	Verbosity(2,
	    fprintf(Of, "\tAtt %s: insufficient cases with known values\n",
			AttName[Att]))
	return;
    }

    GEnv.BrFreq[1] = GEnv.BrFreq[2] = GEnv.BrFreq[3] = 0;

    GEnv.BrSum[1] = GEnv.BrSumSq[1] =
    GEnv.BrSum[2] = GEnv.BrSumSq[2] =
    GEnv.BrSum[3] = GEnv.BrSumSq[3] = 0;

    /*  Omit and count N/A values and count base values  */

    Xp = Fp;
    ForEach(i, Fp, Lp)
    {
	Val = CClass(Case[i]);

	if ( NotApplic(Case[i],Att) )
	{
	    GEnv.BrFreq[1]++;
	    GEnv.BrSum[1]   += Val;
	    GEnv.BrSumSq[1] += Val * Val;

	    Swap(i, Xp);
	    Xp++;
	}
	else
	{
	    GEnv.BrFreq[3]++;
	    GEnv.BrSum[3]   += Val;
	    GEnv.BrSumSq[3] += Val * Val;
	}
    }

    /*  Sort all applicable values  */

    Quicksort(Xp, Lp, Att);

    /*  Try possible cuts between items i and i+1, and determine the
	information and gain of the split in each case  */

    ForEach(i, Xp, Lp - (CMINITEMS*GEnv.FRAC))
    {
	Val = CClass(Case[i]);

	GEnv.BrFreq[2]++;
	GEnv.BrFreq[3]--;

	GEnv.BrSum[2]   += Val;
	GEnv.BrSum[3]   -= Val;
	GEnv.BrSumSq[2] += Val * Val;
	GEnv.BrSumSq[3] -= Val * Val;

	if ( CVal(Case[i+1], Att) > CVal(Case[i], Att) &&
	     i >= Xp+(CMINITEMS*GEnv.FRAC)-1 )
	{
	    ThisGain = ContinGain();
	    if ( ThisGain > BestGain + Epsilon )
	    {
		BestGain = ThisGain;
		BestI    = i;
	    }
	}
    }

    /*  Set the best break point and gain  */ 

    if ( BestGain > Epsilon )
    {
	GEnv.Gain[Att] = BestGain;
	GEnv.Bar[Att]  = Between(CVal(Case[BestI],Att),
				 CVal(Case[BestI+1],Att));

	Verbosity(2,
	    fprintf(Of, "\tAtt %s: cut=%.3f, gain %.3f\n",
			AttName[Att], GEnv.Bar[Att], GEnv.Gain[Att]))

	/*  If not sampling, check subsets now  */

	if ( GEnv.FRAC >= 1.0 )
	{
	    if ( Xp > Fp )
	    {
		NoteTest(Att, 1, GEnv.Bar[Att], Nil);
		FindContinOutliers(Fp, Xp-1, false);
	    }

	    NoteTest(Att, 2, GEnv.Bar[Att], Nil);
	    FindContinOutliers(Fp, BestI, false);

	    NoteTest(Att, 3, GEnv.Bar[Att], Nil);
	    FindContinOutliers(BestI+1, Lp, false);
	}
    }
    else
    {
	Verbosity(2, fprintf(Of, "\tAtt %s: no gain\n", AttName[Att]))
    }
} 



/*************************************************************************/
/*								 	 */
/*    Find the lowest-precision value in the range Low to High		 */
/*								 	 */
/*************************************************************************/


ContValue Between(ContValue Low, ContValue High)
/*	  -------  */
{
    ContValue	Base, Unit, Cut, Try, Margin;

    if ( Low <= 0 && High > 0 ) return 0.0;

    Margin = 0.005L * (High - Low);
    Cut = (Low + High) / 2;

    /*  Try successively smaller units until a threshold lies between
	Low and High  */

    for ( Base = 6 ; Base > -6 ; Base-- )
    {
	Unit = pow(10.0L, Base);
	Try = rint(Cut / Unit) * Unit;

	if ( Try >= Low && Try < High - Margin ) return Try;
	if ( fmod(Low, Unit) < 1E-6 && fmod(High, Unit) < 1E-6 ) break;
    }

    /*  If all else fails, return the low value  */

    return Low;
}



/*************************************************************************/
/*									 */
/*	Set Gain[] for discrete partition of items Fp to Lp		 */
/*									 */
/*************************************************************************/


void CEvalDiscrAtt(Attribute Att, CaseNo Fp, CaseNo Lp)
/*   -------------  */ 
{ 
    if ( MaxAttVal[Att] == 3 )
    {
	EvalBinarySplit(Att, Fp, Lp);
    }
    else
    {
	EvalSubsetSplit(Att, Fp, Lp);
    }

    Verbosity(2,
	if ( GEnv.Gain[Att] > Epsilon )
	{
	    fprintf(Of, "\tAtt %s: gain %.3f\n", AttName[Att], GEnv.Gain[Att]);
	}
	else
	{
	    fprintf(Of, "\tAtt %s: no gain\n", AttName[Att]);
	})
} 



/*************************************************************************/
/*									 */
/*	Special case of binary split					 */
/*									 */
/*************************************************************************/


void EvalBinarySplit(Attribute Att, CaseNo Fp, CaseNo Lp)
/*   ---------------  */ 
{ 
    DiscrValue	v;

    ForEach(v, 1, 3)
    {
	GEnv.BrFreq[v]  = GEnv.DFreq[Att][v][0];
	GEnv.BrSum[v]   = GEnv.DValSum[Att][v];
	GEnv.BrSumSq[v] = GEnv.DValSumSq[Att][v];
    }

    GEnv.Gain[Att] = ContinGain();
    if ( GEnv.Gain[Att] < Epsilon ) GEnv.Gain[Att] = None;

    if ( GEnv.FRAC >= 1 && GEnv.Gain[Att] > Epsilon )
    {
	CheckPotentialClusters(Att, 3, Fp, Lp, 0.0, Nil, Nil);
    }
}



/*************************************************************************/
/*									 */
/*	Divide attribute values into three subsets (one being N/A)	 */
/*									 */
/*************************************************************************/


void EvalSubsetSplit(Attribute Att, CaseNo Fp, CaseNo Lp)
/*   ---------------  */ 
{ 
    DiscrValue	v, sv, Cycle;
    double	ThisGain, BestGain=-1E-6;
    int		Bytes;

    ForEach(v, 1, MaxAttVal[Att])
    {
	GEnv.ValFreq[v]  = GEnv.DFreq[Att][v][0];
	GEnv.ValSum[v]   = GEnv.DValSum[Att][v];
	GEnv.ValSumSq[v] = GEnv.DValSumSq[Att][v];
    }

    GEnv.BrFreq[1]  = GEnv.ValFreq[1];
    GEnv.BrSum[1]   = GEnv.ValSum[1];
    GEnv.BrSumSq[1] = GEnv.ValSumSq[1];

    ForEach(v, 2, 3)
    {
	GEnv.BrFreq[v] = GEnv.BrSum[v] = GEnv.BrSumSq[v] = 0;
    }
   
    ForEach(v, 2, MaxAttVal[Att])
    {
	GEnv.BrFreq[2]  += GEnv.ValFreq[v];
	GEnv.BrSum[2]   += GEnv.ValSum[v];
	GEnv.BrSumSq[2] += GEnv.ValSumSq[v];
    }

    /*  Examine subsets, starting with all values in the left branch.
	At each iteration, move the value with the highest mean from
	the left branch to the right branch and check the gain.
	(In the case of ordered attributes, the value moved is the
	rightmost value in the left branch.)
	Save the best gain so far in Subset[Att]. */

    ForEach(v, 2, MaxAttVal[Att])
    {
	GEnv.Left[v] = ( GEnv.ValFreq[v] > 0 );
    }

    Bytes = (MaxAttVal[Att]>>3) + 1;

    ForEach(Cycle, 2, MaxAttVal[Att])
    {
	if ( Ordered(Att) )
	{
	    for ( sv = MaxAttVal[Att] ; sv > 1 && ! GEnv.Left[sv] ; sv-- )
		;
	}
	else
	{
	    sv = 0;

	    ForEach(v, 2, MaxAttVal[Att])
	    {
		if ( GEnv.Left[v] &&
		     ( ! sv ||
		       GEnv.ValSum[v] / GEnv.ValFreq[v] >
		       GEnv.ValSum[sv] / GEnv.ValFreq[sv] ) )
		{
		    sv = v ;
		}
	    }
	}

	if ( sv < 2 ) break;

	GEnv.Left[sv] = false;

	GEnv.BrFreq[2]  -= GEnv.ValFreq[sv];
	GEnv.BrSum[2]   -= GEnv.ValSum[sv];
	GEnv.BrSumSq[2] -= GEnv.ValSumSq[sv];
	GEnv.BrFreq[3]  += GEnv.ValFreq[sv];
	GEnv.BrSum[3]   += GEnv.ValSum[sv];
	GEnv.BrSumSq[3] += GEnv.ValSumSq[sv];

	if ( GEnv.BrFreq[2] >= (CMINITEMS*GEnv.FRAC) &&
	     GEnv.BrFreq[3] >= (CMINITEMS*GEnv.FRAC) &&
	     (ThisGain = ContinGain()) > BestGain + Epsilon )
	{
	    GEnv.Gain[Att] = BestGain = ThisGain;
	    GEnv.Bar[Att]  = sv-1;

	    /*  Record in Subset[Att]  */

	    ClearBits(Bytes, GEnv.Subset[Att]);

	    ForEach(v, 2, MaxAttVal[Att])
	    {
		if ( GEnv.Left[v] )
		{
		    SetBit(v, GEnv.Subset[Att]);
		}
	    }
	}
    }

    if ( GEnv.FRAC >= 1 && GEnv.Gain[Att] > Epsilon )
    {
	CheckPotentialClusters(Att, 3, Fp, Lp, GEnv.Bar[Att], GEnv.Subset[Att],
			       Nil);
    }
}



double SDEstimate(CaseCount N, double Sum, double SumSq)
/*     ----------  */
{
    return sqrt( (SumSq - Sum * Sum / N + 1E-3) / (N - 1) );
}



/*************************************************************************/
/*								  	 */
/*	Compute continuous gain for three branches			 */
/*								  	 */
/*************************************************************************/


double ContinGain()
/*     ----------  */
{
    double	Resid=0;
    DiscrValue	v;
    CaseCount	Cases=0;

    ForEach(v, 1, 3)
    {
	if ( GEnv.BrFreq[v] > 1 )
	{
	    Cases += GEnv.BrFreq[v];
	    Resid += GEnv.BrFreq[v] *
		     SDEstimate(GEnv.BrFreq[v], GEnv.BrSum[v], GEnv.BrSumSq[v]);
	}
    }

    return GEnv.PSD - Resid / Cases;
}
