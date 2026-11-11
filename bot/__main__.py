from tri.runtime import cap_threads

cap_threads()            # v3.8: BLAS pools sized to the CPU quota, before numpy loads

from .discord_bot import main  # noqa: E402

main()
