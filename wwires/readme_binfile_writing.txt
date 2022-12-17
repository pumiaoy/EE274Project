Raw Data Header
=====================================================


U32: tag = 0 for header length
U32: 4 bytes
U32: Total header length

U32: tag = 1 for time
U32: 12 bytes
U32: 1904 for base year
U32:  0      This zero allows these last two U32's to be read as an I64 because Java cannot read U32's
U32:  Seconds since January 1st, 1904 12 AM, universal time

U32: tag = 2 for User Header
U32: size of header in bytes
U32: User Header. It is forced to be an even number of bytes.

U32: tag = 3 for format
U32: 4 bytes
U32: Format = 1 for 25% compression.  See end of spec for details.

U32: tag = 4 for electrode array
U32: 8 bytes
U32: number of channels
U32: array #   500 and up are 512 arrays.

U32: tag = 5 for frequency in Hz
U32: 4 bytes
U32: Frequency in Hz

U32: tag = 6 for trigger parameters
U32: 8 bytes  This was a 12 by mistake for the December trip
U32: First Trigger Delay in seconds * 1000000
U32: First Trigger Interval in seconds * 1000000

U32: tag = 7 for the data set identifier
U32: length l (total length of the data set identifier)
var: utf-8 string of length l bytes

U32: Tag = 499 for end of header.
U32: 4 bytes
U32:  Number of seconds * Acq Freq



Followed by the data.
=====================================================

We compress the 16 bit (big endian) data into 12 bits by throwing away the most significant 4 bits.  These are not used by the DAC.  To pack, the algorithm is:
Copy channel 0 (TTL) as is.
Add 2048 to all other I16s.  Convert channel 1-2, 3-4, 5-6 etc. into three bytes each.
b1= s1 >> 4;
b2= (s1 & 0x000F) << 4 + s2 >>8;
b3=s2 & 0x00FF;

Do this for the whole sample.  Repeat for each sample.

The unpack algorithm is the following:

Copy bytes 0,1 directly into the TTL signal (channel 0)
Convert bytes 2-4, 5-7, 8-10 etc. into two samples each:
s1=(b1<<4) + (b2 >>4);
s2=(b2 & 0x000F) << 8 + b3;

Subtract 2048 from all channels except zero.
Continue converting each set of 770 bytes into 513 electrodes until done.



This is different in 519 array data. Channel 0 is no longer treated specially.
Add 2048 to all the channels. Convert channel 0-1, 2-3, 4-5 etc. into three bytes each.

To unpack, convert bytes 0-2, 3-5, 6-8 etc. into two samples each.
Subtract 2048 from all the channels.
Continue converting each set of 780 bytes into 520 electrodes until done.
