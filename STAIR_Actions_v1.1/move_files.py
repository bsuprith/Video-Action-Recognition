"""
After extracting the RAR, we run this to move all the files into
the appropriate train/test folders.

Should only run this file once!
"""
import os
import os.path

def get_train_test_lists():
    """
    Using one of the train/test files (01, 02, or 03), get the filename
    breakdowns we'll later use to move everything.
    """
    # Get our files based on version.
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    test_file = os.path.join(__location__, 'videolist_val.txt')
    train_file = os.path.join(__location__, 'videolist_train.txt')

    # Build the test list.
    with open(test_file) as fin:
        test_list = [row.strip() for row in list(fin)]
        test_list = [row.split(' ')[0] for row in test_list]

    # Build the train list. Extra step to remove the class index.
    with open(train_file) as fin:
        train_list = [row.strip() for row in list(fin)]
        train_list = [row.split(' ')[0] for row in train_list]

    # Set the groups in a dictionary.
    file_groups = {
        'train': train_list,
        'test': test_list
    }

    return file_groups

def move_files(file_groups):
    """This assumes all of our files are currently in _this_ directory.
    So move them to the appropriate spot. Only needs to happen once.
    """
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # Do each of our groups.
    for group, videos in file_groups.items():

        # Do each of our videos.
        for video in videos:

            # Get the parts.
            parts = video.split("/")
            classname = parts[0]
            filename = parts[1]

            # Check if this class exists.
            if not os.path.exists(os.path.join(__location__,group, classname)):
                print("Creating folder for %s/%s" % (group, classname))
                os.makedirs(os.path.join(__location__,group, classname))

            # Check if we have already moved this file, or at least that it
            # exists to move.
            if not os.path.exists(os.path.join(__location__,classname, filename)):
                print("Can't find %s to move. Skipping." % (filename))
                continue

            # Move it.
            dest = os.path.join(__location__,group, classname, filename)
            source = os.path.join(__location__,classname, filename)
            print("Moving %s to %s" % (filename, dest))
            os.rename(source, dest)

    print("Done.")

def main():
    """
    Go through each of our train/test text files and move the videos
    to the right place.
    """
    # Get the videos in groups so we can move them.
    group_lists = get_train_test_lists()

    # Move the files.
    move_files(group_lists)

if __name__ == '__main__':
    main()
