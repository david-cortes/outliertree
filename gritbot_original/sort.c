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
/*	Sorting utilities						 */
/*	-----------------						 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"

#define SwapPair(a,b)	{Xab=Pair[a]; Pair[a]=Pair[b]; Pair[b]=Xab;}

SortPair	*Pair=Nil;


/*************************************************************************/
/*									 */
/*	Sort items from Fp to Lp on attribute Att.			 */
/*	To maximise cache hits, values are copied into Pair and		 */
/*	the results copied back to Case.				 */
/*									 */
/*************************************************************************/


void Quicksort(CaseNo Fp, CaseNo Lp, Attribute Att)
/*   ---------  */
{
    CaseNo i;

    ForEach(i, Fp, Lp)
    {
	Pair[i].C = CVal(Case[i], Att);
	Pair[i].D = Case[i];
    }

    Cachesort(Fp, Lp);

    ForEach(i, Fp, Lp)
    {
	Case[i] = Pair[i].D;
    }
}



/*************************************************************************/
/*									 */
/*	Sort elements Fp to Lp of Pair					 */
/*									 */
/*************************************************************************/


void Cachesort(CaseNo Fp, CaseNo Lp)
/*   ---------  */
{
    CaseNo	i, Middle, High;
    ContValue	Thresh, Val;
    SortPair	Xab;

    while ( Fp < Lp )
    {
	Thresh = Pair[(Fp+Lp) / 2].C;

	/*  Divide elements into three groups:
		Fp .. Middle-1: values < Thresh
		Middle .. High: values = Thresh
		High+1 .. Lp:   values > Thresh  */

	for ( Middle = Fp ; Pair[Middle].C < Thresh ; Middle++ )
	    ;

	for ( High = Lp ; Pair[High].C > Thresh ; High-- )
	    ;

	for ( i = Middle ; i <= High ; )
	{
	    if ( (Val = Pair[i].C) < Thresh )
	    {
		SwapPair(Middle, i);
		Middle++;
		i++;
	    }
	    else
	    if ( Val > Thresh )
	    {
		SwapPair(High, i);
		High--;
	    }
	    else
	    {
		i++;
	    }
	}

	/*  Sort the first group  */

	Cachesort(Fp, Middle-1);

	/*  Continue with the last group  */

	Fp = High+1;
    }
}
