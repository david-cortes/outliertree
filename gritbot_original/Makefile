#*************************************************************************#
#*									 *#
#*		Makefile for GritBot					 *#
#*		--------------------					 *#
#*									 *#
#*************************************************************************#


CC     = gcc -ffloat-store
CFLAGS = -DVerbOpt -g -Wall -O0
LFLAGS = $(S)
SHELL  = /bin/csh


#	Definitions of file sets

src =\
	global.c\
	cluster.c\
	continatt.c\
	outlier.c\
	getdata.c\
	gritbot.c\
	sort.c\
	discratt.c\
	check.c\
	common.c\
	getnames.c\
	implicitatt.c\
	modelfiles.c\
	update.c\
	utility.c


isrc =\
	inspect.c\
	cluster.c\
	outlier.c\
	getdata.c\
	getnames.c\
	implicitatt.c\
	modelfiles.c\
	update.c\
	common.c\
	utility.c


obj =\
	global.o\
	gritbot.o\
	getdata.o getnames.o implicitatt.o\
	check.o cluster.o outlier.o\
	common.o continatt.o discratt.o\
	modelfiles.o\
	sort.o utility.o update.o


all:
	make gritbot
	make inspect


# debug version (including verbosity option)

gritbotdbg:\
	$(obj) defns.i text.i extern.i Makefile
	$(CC) -DVerbOpt -g -o gritbotdbg $(obj) -lm

inspectdbg:\
	$(isrc) defns.i text.i Makefile
	cat defns.i $(isrc)\
		| egrep -v 'defns.i|extern.i' >insgt.c
	$(CC) $(CFLAGS) -DVerbOpt -DINSPECT -o inspectdbg insgt.c -lm

# production versions

gritbot:\
	$(src) defns.i text.i Makefile
	cat defns.i $(src)\
		| egrep -v 'defns.i|extern.i' >gbotgt.c
	$(CC) $(LFLAGS) -O3 -o gritbot gbotgt.c -lm
	strip gritbot
	rm gbotgt.c

inspect:\
	$(isrc) defns.i text.i Makefile
	cat defns.i $(isrc)\
		| egrep -v 'defns.i|extern.i' >insgt.c
	$(CC) $(LFLAGS) -DINSPECT -O3 -o inspect insgt.c -lm
	strip inspect
	rm insgt.c


$(obj):		Makefile defns.i extern.i


.c.o:
	$(CC) $(CFLAGS) -c $<
