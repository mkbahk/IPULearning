parallel --keep-order '''
  ping -c 1 -W 1 10.1.1.{} >/dev/null 2>&1 && bmc="\e[00;32m✓\e[0m" || bmc="\e[00;31mX\e[0m"
  ping -c 1 -W 1 10.1.2.{} >/dev/null 2>&1 && gw="\e[00;32m✓\e[0m" || gw="\e[00;31mX\e[0m"
  ping -c 1 -W 1 10.1.5.{} >/dev/null 2>&1 && mx="\e[00;32m✓\e[0m" || mx="\e[00;31mX\e[0m"
  version=$(sshpass -p ChangeMeFdh5P ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR itadmin@10.1.2.{} jq -r '.version' /usr/share/ipum-support/manifest.json 2>/dev/null)
  printf "ipum%2d bmc:[10.1.1.{} %b ] gw:[10.1.2.{} %b ] mx:[ 10.1.5.{} %b ] manifest: [ %s ]\n" {} "$bmc" "$gw" "$mx" $version
''' ::: {1..4}
