
# ============================================================
# TPU MEMORY FILE COPY SCRIPT
# Automatically runs before Vivado synthesis.
# ============================================================

# ------------------------------------------------------------
# FIND PROJECT AND SYNTHESIS DIRECTORIES
# ------------------------------------------------------------

# Find the directory containing this Tcl script.
set project_dir [file dirname [file normalize [info script]]]

# Source memory directory.
set source_mem_dir [file join $project_dir "mem_files"]

# ------------------------------------------------------------
# FIND THE ACTUAL VIVADO RUN DIRECTORY
# ------------------------------------------------------------

# Search for the .runs directory instead of assuming its name.
set run_dirs [glob -nocomplain -directory $project_dir "*.runs"]

# Keep only directories.
set valid_run_dirs {}

foreach directory $run_dirs {
    if {[file isdirectory $directory]} {
        lappend valid_run_dirs $directory
    }
}

# Require exactly one Vivado runs directory.
if {[llength $valid_run_dirs] != 1} {
    error "ERROR: Expected exactly one .runs directory, found [llength $valid_run_dirs]: $valid_run_dirs"
}

# Find the synthesis directory.
set synth_dir [file join [lindex $valid_run_dirs 0] "synth_1"]

# Destination memory directory.
set dest_mem_dir [file join $synth_dir "mem_files"]

puts "\n================================================"
puts "TPU MEMORY FILE SETUP"
puts "================================================"

puts "Project:     $project_dir"
puts "Source:      $source_mem_dir"
puts "Destination: $dest_mem_dir"

# ------------------------------------------------------------
# CHECK SOURCE MEMORY DIRECTORY
# ------------------------------------------------------------

if {![file isdirectory $source_mem_dir]} {
    error "ERROR: Project memory directory does not exist: $source_mem_dir"
}

# ------------------------------------------------------------
# RECURSIVELY COLLECT ALL .MEM FILES
# ------------------------------------------------------------

proc find_mem_files {directory} {
    set results {}

    foreach item [glob -nocomplain -directory $directory *] {
        if {[file isdirectory $item]} {
            set nested_results [find_mem_files $item]
            set results [concat $results $nested_results]
        } elseif {[string equal -nocase [file extension $item] ".mem"]} {
            lappend results $item
        }
    }

    return $results
}

# Count source memory files.
set source_files [find_mem_files $source_mem_dir]
set source_count [llength $source_files]

puts "\nSource memory files found: $source_count"

if {$source_count == 0} {
    error "ERROR: No .mem files found in the project memory directory."
}

# ------------------------------------------------------------
# CHECK SYNTHESIS DIRECTORY
# ------------------------------------------------------------

if {![file isdirectory $synth_dir]} {
    error "ERROR: Synthesis directory does not exist: $synth_dir"
}

# ------------------------------------------------------------
# COPY MEMORY FILES INTO SYNTHESIS DIRECTORY
# ------------------------------------------------------------

if {[file exists $dest_mem_dir]} {
    file delete -force $dest_mem_dir
}

file copy -force $source_mem_dir $synth_dir

# ------------------------------------------------------------
# VERIFY MEMORY FILE COPY
# ------------------------------------------------------------

set dest_files [find_mem_files $dest_mem_dir]
set dest_count [llength $dest_files]

puts "Synthesis memory files found: $dest_count"

if {$source_count != $dest_count} {
    error "ERROR: Memory file count mismatch! Source=$source_count Destination=$dest_count"
}

# ------------------------------------------------------------
# PRINT COPIED MEMORY FILES
# ------------------------------------------------------------

puts "\nCopied memory files:"

foreach mem_file $dest_files {
    puts "  $mem_file"
}

puts "\n================================================"
puts "SUCCESS: TPU MEMORY FILES COPIED AND VERIFIED"
puts "================================================\n"
