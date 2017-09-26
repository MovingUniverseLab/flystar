from astropy.table import Table


class StarTable(object):
    
    """
    A startable is an astropy.Table collecting informations on matched stars
    across multiple starlists.
    """

    def __init__(self, name, x, xe, y, ye, m, me, ref_list=0,
                 file_list=list()):
        
        Table.__init__(self)
        self.names = ('name', 'x', 'xe', 'y', 'ye', 'm', 'me')
#        self = Table([name, x, xe, y, ye, m, me], names=('name', 'x', 'xe',
 #                    'y', 'ye', 'm', 'me'), meta={'ref_list': ref_list,
  #                   'file_list': file_list})
