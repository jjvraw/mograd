from std.sys import get_defined_int

# Set via -D MOGRAD_DEBUG=1 (op names) or -D MOGRAD_DEBUG=2 (op names + shapes)
comptime DEBUG = get_defined_int["MOGRAD_DEBUG", 0]()
