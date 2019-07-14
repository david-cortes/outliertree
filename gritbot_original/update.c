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
/*		Routines that provide information on progress		 */
/*              ---------------------------------------------		 */
/*									 */
/*************************************************************************/


#include "defns.i"
#include "extern.i"


int	Stage=0;		/* Current stage number  */
FILE	*Uf=0;			/* File to which update info written  */


/*************************************************************************/
/*									 */
/*	There are five stages (see messages in Progress() below)	 */
/*	Record stage and open update file if necessary			 */
/*									 */
/*************************************************************************/


void NotifyStage(int S)
/*   -----------  */
{
    Stage = S;
    if ( S == 1 )
    {
	if ( ! (Uf = GetFile(".tmp", "w")) ) Error(NOFILE, "", " for writing");
    }
}



/*************************************************************************/
/*									 */
/*	Print progress message.  This routine is called in two ways:	 */
/*	  *  negative Delta = measure of total effort required for stage */
/*	  *  positive Delta = increment since last call			 */
/*									 */
/*************************************************************************/


void Progress(int Delta)
/*   --------  */
{
    static int   Att, Current=0, Twentieth=0, LastStage=0;
    int		 p;
    static char *Message[]={ "",
			     T_ReadTrain,
			     T_ReadTest,
			     T_Prelim,
			     T_Checking,
			     T_Reporting,
			     T_CleaningUp },
		*Done=">>>>>>>>>>>>>>>>>>>>",
		*ToDo="....................";

    if ( ! Uf ) return;

    if ( Delta < 0)
    {
	Att       = -Delta;
	Current   = 0;
	Twentieth = -1;

	if ( Stage == PRELIM )
	{
	    fprintf(Uf, F_Preliminaries, AttName[Att]);
	    fflush(Uf);
	}
    }
    else
    {
	Current = Min(MaxCase+1, Current + Delta);
    }

    if ( Stage != PRELIM &&
	 ( (p = rint((20.0 * Current) / (MaxCase+1.01))) != Twentieth ||
	   Stage != LastStage ) )
    {
	LastStage = Stage;
	Twentieth = p;

	if ( Stage == CHECKING )
	{
	    fprintf(Uf, F_Checking(AttName[Att],
				   Done + (20 - Twentieth), ToDo + Twentieth,
				   Current));
	}
	else
	{
	    fprintf(Uf, "%s\n", Message[Stage]);
	}

	fflush(Uf);
    }
}
