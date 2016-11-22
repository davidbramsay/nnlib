TARGET=nn

CFLAGS=-g -Wall
CC=gcc

ODIR=obj

DEPS =nn.h matrix.h
SOURCES=nn.c matrix.c

OBJECTS=$(addprefix $(ODIR)/,$(SOURCES:.c=.o))


$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f *.o *.a $(ODIR)/* $(TARGET)

