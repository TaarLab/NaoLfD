import os

def fix_csv(input_file, output_file, num_fields):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    with open(output_file, 'w') as outfile:

        fields = lines[0].strip().split(',')

        if len(fields) == num_fields:
            return

        for i in range(0, len(fields), num_fields):
            outfile.write(','.join(fields[i:i + num_fields]) + '\n')

        for line in lines[1:]:
            outfile.write(line)

def fix_dataset_csvs(dataset_dir, num_fields):
    for root, dirs, files in os.walk(dataset_dir):
        for name in files:
            if name == 'data.csv':
                input_file_path = os.path.join(root, name)
                output_file_path = os.path.join(root, 'data.csv')  # Change this if needed
                fix_csv(input_file_path, output_file_path, num_fields)
                print(f"Fixed CSV file at: {output_file_path}")

dataset_directory = r'.\NaoPickAndPlaceData'
number_of_fields_per_record = 40

fix_dataset_csvs(dataset_directory, number_of_fields_per_record)
