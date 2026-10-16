"""Speaker marking: who said each part of the transcript (doctor / patient or family / unknown / background voices). See SPEC 4.3."""
DOCTOR, OTHER, UNKNOWN, BACKGROUND = 'doctor', 'other', 'unknown', 'background'
LABEL_NAMES = {DOCTOR: '醫師', OTHER: '患者或家屬', UNKNOWN: '不明', BACKGROUND: '背景'}
BACKGROUND_GID = -1       # the "voice group" `SpeakerTracker.voice_groups_for` reports for a piece of background voices
TEXT_SOURCES = ('text', 'text2')      # labels read from the text (first / second text fill), not heard: shown with a star
