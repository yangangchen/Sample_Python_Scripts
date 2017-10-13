# renameFiles.py
# 
# Copyright (C) 2017  Yangang Chen
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

################################

import re, glob, os


def renameFiles(files, pattern, replacement):
    for old_pathandfilename in glob.glob(files):
        path = os.path.dirname(old_pathandfilename)
        old_filename = os.path.basename(old_pathandfilename)
        new_filename = re.sub(pattern, replacement, old_filename)
        # print(old_filename)
        # print(new_filename)
        if new_filename != old_filename:
            # input("debug stop")
            os.rename(old_pathandfilename, os.path.join(path, new_filename))


################################

renameFiles('Figures/*', r'^IMG_(.*).JPG$', r'img_\1.jpg')
