"""Input validation service for checkpoint resumption."""

import re
from typing import Dict, Any, Optional, Tuple


class InputValidationError(Exception):
    """Exception raised when user input validation fails."""
    pass


class InputValidator:
    """Validates user input before resuming workflows."""

    @staticmethod
    def validate_year_input(
        user_input: Dict[str, Any],
        available_years: Dict[str, Dict[str, int]]
    ) -> Tuple[bool, Optional[Dict[str, int]], Optional[str]]:
        """
        Validate user-provided year input.

        Args:
            user_input: Dict with "year" field
            available_years: Available year ranges by category

        Returns:
            Tuple of (is_valid, parsed_years, error_message)
            parsed_years: {"min": int, "max": int} if valid
        """
        if "year" not in user_input:
            return False, None, "No year provided"

        year_input = user_input["year"]

        # Security: Validate input type
        if not isinstance(year_input, (int, str)):
            return False, None, "Invalid year format - must be number or text"

        # Convert to string for parsing
        year_str = str(year_input).strip()

        # Security: Check for SQL injection patterns
        dangerous_patterns = [
            r"(union|select|insert|update|delete|drop|create|alter|exec|execute)",
            r"(--|;|\/\*|\*\/|xp_|sp_)",
            r"(<script|javascript:|onerror|onclick)",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, year_str, re.IGNORECASE):
                return False, None, "Invalid characters detected in year input"

        # Security: Limit input length
        if len(year_str) > 50:
            return False, None, "Year input too long"

        # Extract years using regex (same as verification agent)
        year_patterns = [
            r'\b(19\d{2}|20[0-4]\d)\b',  # Single year
            r'\b(19\d{2}|20[0-4]\d)\s*(?:to|-|through|until)\s*(19\d{2}|20[0-4]\d)\b',  # Range
        ]

        years_found = []
        for pattern in year_patterns:
            matches = re.findall(pattern, year_str, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        years_found.extend([int(y) for y in match if y])
                    else:
                        years_found.append(int(match))

        if not years_found:
            return False, None, f"Could not parse year from '{year_str}'. Please provide a valid year (e.g., 2020 or 2018-2022)"

        # Parse year range
        min_year = min(years_found)
        max_year = max(years_found)
        parsed_years = {"min": min_year, "max": max_year}

        # Validate year is reasonable (not too far in past/future)
        if min_year < 1900 or max_year > 2100:
            return False, None, f"Year {min_year}-{max_year} is out of reasonable range (1900-2100)"

        # Check if year is available in any category
        if available_years:
            year_available = False
            for category, year_range in available_years.items():
                cat_min = year_range["min"]
                cat_max = year_range["max"]

                # Check overlap
                if min_year <= cat_max and max_year >= cat_min:
                    year_available = True
                    break

            if not year_available:
                # Format available years message
                year_info = []
                for cat, yr in available_years.items():
                    year_info.append(f"{cat.replace('_', ' ').title()}: {yr['min']}-{yr['max']}")

                error_msg = (
                    f"We don't have data for {min_year}-{max_year}. "
                    f"Available years:\n" + "\n".join(f"• {info}" for info in year_info)
                )
                return False, None, error_msg

        # All validations passed
        return True, parsed_years, None


def get_input_validator() -> InputValidator:
    """Get the input validator instance."""
    return InputValidator()
