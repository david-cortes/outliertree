AC_INIT([outliertree], 1.7.5) 
AC_LANG(C++)
AC_TRY_COMPILE(
    [],
    [double *__restrict x;],
    [SUPPORTS_RESTRICT="-DSUPPORTS_RESTRICT=1"],
    [SUPPORTS_RESTRICT=""]
)
AC_SUBST(SUPPORTS_RESTRICT)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT 
