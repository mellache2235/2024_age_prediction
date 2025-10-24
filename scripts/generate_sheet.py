import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font

# Create a new workbook
wb = Workbook()

# -----------------------------
# Sheet 1: Programs (with Faculty column)
# -----------------------------
ws_programs = wb.active
ws_programs.title = "Programs"

# Define headers for the Programs sheet
headers_programs = [
    "University/Institution", "Country", "City", "Program/Department",
    "Research Focus/Areas", "Coursework & Program Structure",
    "Language of Instruction", "Funding Opportunities", "Application Deadlines",
    "Admission Requirements", "Faculty", "Website/Contact Info", "Additional Notes"
]
ws_programs.append(headers_programs)

# Set column widths for Programs sheet for better readability
column_widths_programs = {
    "A": 30,   # University/Institution
    "B": 15,   # Country
    "C": 15,   # City
    "D": 30,   # Program/Department
    "E": 40,   # Research Focus/Areas
    "F": 40,   # Coursework & Program Structure
    "G": 25,   # Language of Instruction
    "H": 30,   # Funding Opportunities
    "I": 25,   # Application Deadlines
    "J": 35,   # Admission Requirements
    "K": 40,   # Faculty (list multiple names if needed)
    "L": 40,   # Website/Contact Info
    "M": 50,   # Additional Notes
}

for col, width in column_widths_programs.items():
    ws_programs.column_dimensions[col].width = width

# Format the header row for Programs (bold, wrapped text, vertically centered)
for cell in ws_programs[1]:
    cell.alignment = Alignment(wrap_text=True, vertical="center")
    cell.font = Font(bold=True)

# Example data row for Programs sheet
ws_programs.append([
    "Example University", "USA", "Boston", "Computational Neuroscience PhD",
    "Neural coding, brain-computer interfaces", "Core courses + lab rotations",
    "English", "Full funding via RA/TA positions", "Dec 15, 2025",
    "GRE, TOEFL, Bachelor in Neuroscience", "Dr. Jane Doe, Dr. John Smith",
    "http://www.example.edu/phd-compneuro", "High living cost; strong industry ties"
])

# -----------------------------
# Sheet 2: Faculty
# -----------------------------
ws_faculty = wb.create_sheet("Faculty")

# Define headers for the Faculty sheet
headers_faculty = [
    "University/Institution", "Program/Department", "Faculty Name", "Faculty Title",
    "Research Interests", "Email", "Website/Contact Info", "Additional Notes"
]
ws_faculty.append(headers_faculty)

# Set column widths for Faculty sheet for better readability
column_widths_faculty = {
    "A": 30,  # University/Institution
    "B": 30,  # Program/Department
    "C": 25,  # Faculty Name
    "D": 20,  # Faculty Title
    "E": 40,  # Research Interests
    "F": 30,  # Email
    "G": 40,  # Website/Contact Info
    "H": 50,  # Additional Notes
}

for col, width in column_widths_faculty.items():
    ws_faculty.column_dimensions[col].width = width

# Format the header row for Faculty (bold, wrapped text, vertically centered)
for cell in ws_faculty[1]:
    cell.alignment = Alignment(wrap_text=True, vertical="center")
    cell.font = Font(bold=True)

# Example data row for Faculty sheet
ws_faculty.append([
    "Example University", "Computational Neuroscience PhD", "Dr. Jane Doe",
    "Professor", "Computational modeling of neural circuits", "jdoe@example.edu",
    "http://www.example.edu/faculty/jdoe", "Accepting new PhD students"
])

# -----------------------------
# Save the workbook to a file
# -----------------------------
output_filename = "PhD_Programs_and_Faculty_Template.xlsx"
wb.save(output_filename)
print(f"Excel template saved as '{output_filename}'")

