from database import db
class Name(db.Model):

    __tablename__ = 'names'

    id = db.Column(db.Integer, primary_key=True)
    lastname = db.Column(db.String(50), nullable=True)
    hispanic = db.Column(db.Float, nullable=True)
    asian = db.Column(db.Float, nullable=True)
    white = db.Column(db.Float, nullable=True)
    african = db.Column(db.Float, nullable=True)
