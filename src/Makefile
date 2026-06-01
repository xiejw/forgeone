CFLAGS = -std=c23 -Wall -Werror -pedantic -Wextra -Wfatal-errors -Wconversion
ifdef RELEASE
  CFLAGS += -O2 -DNDEBUG -march=native -flto -ffast-math
endif

BUILD   = .build
LIB_SRC = env.c policy.c runner.c
LDLIBS  = -lm

all: $(BUILD)/cart

$(BUILD)/cart: $(LIB_SRC) main.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $(LIB_SRC) main.c $(LDLIBS)

test: $(BUILD)/test
	$(BUILD)/test

$(BUILD)/test: $(LIB_SRC) test.c | $(BUILD)
	$(CC) $(CFLAGS) -D_POSIX_C_SOURCE=200809L -o $@ $(LIB_SRC) test.c $(LDLIBS)

$(BUILD):
	mkdir -p $(BUILD)

run: all
	$(BUILD)/cart

run_random: all
	$(BUILD)/cart random

run_rev: all
	$(BUILD)/cart rev

clean:
	rm -rf $(BUILD)

.PHONY: all test run run_random run_rev clean fmt

fmt:
	~/Workspace/y/tools/scripts/clang_format_all.sh .
