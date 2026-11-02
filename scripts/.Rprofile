# Load the project profile while keeping scripts/ as the working directory.
local({
  previous_directory <- setwd("..")
  tryCatch(
    sys.source(".Rprofile", envir = .GlobalEnv),
    finally = setwd(previous_directory)
  )
})
