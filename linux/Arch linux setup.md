This is a record of everything that I did to get arch linux up and running on virtualbox

Note: if running in virtualbox, enable efi by going to Settings -> system -> check enable EFI

## partitioning the disks

```zsh
fdsik -l
```
This will list available disks

modify partition tables: 
```zsh
fdsik /dev/<disk>
```

this enters into <> (whatever that tools to make partition tables is called)

- make a new partition: `n`
- make 3 partitions In total
- EFI partition
	- enter `1` for partition number 
	- enter `2048` for first sector
	- enter `+300M` for last sector (makes a partition of 300mb) (this is the minimum recommended by the installation guide)
- swap partition
	- partition number `2` (use default)
	- use default for first sector
	- enter `+512M` for last sector
- root partition
	- default values for partition number and first sector
	- use default value for last sector to use the rest of the available disk
- hit `w` to save changes


## Format the partitions

running `fdisk -l` should now give you a list of the partiions you just made. This is what the output of the command was in my case:
![[Pasted image 20221219171627.png]]


- format the root partition: `mkfs.ext4 /dev/sda3`
- format the swap partition: `mkswap /dev/sda2`
- format the efi partition: `mkfs.fat -F 32 /dev/sda1`
	- MAKE SURE YOU HEED THE WARNING ON THE ARCH INSTALLATION GUIDE
	- FORMATTING THIS MAY DESTROY THE BOOTLOADERS FOR OTHER OSs


## Mount filesystems

- root: `mount /dev/sda3 /mnt`
- efi: `mount --mkdir /dev/sda1 /mnt/boot`
	- bugfix: trying `mount --mkdir /dev/sda1 /mnt/boot/efi`
- swap: `swapon /dev/sda2`


## install base packages and configuration
`pacstrap -K /mnt base linux linux-firmware vim nano git`

- If you get an error about an invalid or corrupted package (pgp signature), try reinstalling archlinux-keyring (`sudo pacman -Sy archlinux-keyring`) (https://ostechnix.com/fix-invalid-corrupted-package-pgp-signature-error-arch-linux/)

**configure the sytem**
- generate an fstab file
	- fstab tells systemd how disk partitions should be mounted into the filesystem
	- generate using `genfstab -U /mnt >> /mnt/etc/fstab`'
	- it should be automatically populated with the correct partitions

**change root into the systeminvalid
- `arch-chroot /mnt`

**set time zone**
`ln -sf /usr/share/zoneinfo/<Region>/<City> /etc/localtime`
`ls` through that zoneinfo folder to see the available options

now run:
`hwclock --systohc`


**localization**
vim into `/etc/locale.gen` and uncomment `en_US.UTF-8 UTF-8` `

generate locales by running:
`locale-gen`


create the file `/etc/locale.conf` and in that file set `LANG=en_US.UTF-8`

**password**
set the root password using `passwd`


**bootloader**
`pacman -S grub efibootmgr`
`grub-install --target=x86_64-efi --efi-directory=/boot --bootloader-id=GRUB`

## Setting up a desktop

install video driver:(?)
`sudo pacman -Ss xf86-video-intel`

install plasma
`sudo pacman -Sy plasma`

install kde-applications
`sudo pacman -Sy kde-applications`

install ssdm (login display)
`sudo pacman -Sy sddm`

enable display manager and network manager:
`systemctl enable sddm.service`
`systemctl enable NetworkManager.service`

# TODO

made some fixes based on this video: https://www.youtube.com/watch?v=rUEnS1zj1DM

need to update doc accordingly

# References

- https://wiki.archlinux.org/title/Installation_guide
- https://itsfoss.com/install-kde-arch-linux/
- https://www.youtube.com/watch?v=rUEnS1zj1DM