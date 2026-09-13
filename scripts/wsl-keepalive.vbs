Set sh = CreateObject("Wscript.Shell")
sh.Run "wsl -d Ubuntu --exec tail -f /dev/null", 0

